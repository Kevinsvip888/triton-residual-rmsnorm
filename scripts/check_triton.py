#!/usr/bin/env python3
"""Smoke-check the first fused Triton kernel; no speed measurements."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache/triton'))
sys.path.insert(0, str(ROOT))

import torch
from rmsnorm_lab.reference import residual_rmsnorm_reference
from rmsnorm_lab.triton_kernel import residual_rmsnorm_triton, launch_config
from check_reference import oracle
from check_compiled_reference import check_outputs


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require(torch.cuda.is_available(), 'CUDA unavailable')
    require(torch.cuda.is_bf16_supported(), 'BF16 unavailable')
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = json.loads(contract_path.read_text())
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Unexpected contract')
    previous = []
    for p in (ROOT / 'outputs/day8').glob('*/compiled_reference_checks.json'):
        receipt = json.loads(p.read_text())
        if (receipt.get('contract_sha256') == sha(contract_path) and len(receipt.get('cases', [])) == 8
                and all(sha(ROOT / path) == digest for path, digest in receipt['file_sha256'].items())):
            previous.append(p)
    require(previous, 'Run the compiled-reference checks successfully with the current files first')
    torch.manual_seed(44)
    eps, records = contract['eps'], []
    shapes = ((1, 1), (1, 768), (32, 1024), (3, 2560), (32, 4096),
              (17, 257), (2, 4097), (2, 8192), (2048, 768))
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for dtype, label, half_ulp in ((torch.float16, 'float16', 2**-11),
                                      (torch.bfloat16, 'bfloat16', 2**-8)):
            cases = [('random', m, h) for m, h in shapes] + [
                ('halfway', 1, 1024), ('zero', 2, 257), ('hand', 1, 4)]
            for pattern, m, h in cases:
                if pattern == 'random':
                    x = torch.randn(m, h, device='cuda', dtype=torch.float32).to(dtype)
                    residual = torch.randn(m, h, device='cuda', dtype=torch.float32).to(dtype)
                    weight = torch.randn(h, device='cuda', dtype=torch.float32).to(dtype)
                elif pattern == 'halfway':
                    x = torch.ones(m, h, device='cuda', dtype=dtype)
                    residual = torch.zeros_like(x)
                    residual[:, 0] = half_ulp
                    weight = torch.ones(h, device='cuda', dtype=dtype)
                elif pattern == 'zero':
                    x = torch.zeros(m, h, device='cuda', dtype=dtype)
                    residual = torch.zeros_like(x)
                    weight = torch.ones(h, device='cuda', dtype=dtype)
                else:
                    x = torch.tensor([[1., 0., 0., 0.]], device='cuda', dtype=dtype)
                    residual = x.clone()
                    weight = torch.tensor([1., 2., -1., .5], device='cuda', dtype=dtype)
                name = f'{label}/{pattern}/{m}x{h}'
                print('Checking ' + name + ' (new signatures may compile)...', flush=True)
                inputs = (x, residual, weight)
                saved_inputs = [v.clone() for v in inputs]
                expected = oracle(x, residual, weight, eps)
                eager = residual_rmsnorm_reference(x, residual, weight, eps)
                tolerances = contract['output_tolerances'][label]
                actual = residual_rmsnorm_triton(x, residual, weight, eps)
                error = check_outputs(actual, expected, inputs, tolerances, name + '/oracle')
                check_outputs(actual, eager, inputs, tolerances, name + '/eager')
                snapshots = [v.clone() for v in actual]
                repeated = residual_rmsnorm_triton(x, residual, weight, eps)
                check_outputs(repeated, expected, inputs, tolerances, name + '/repeated')
                require(not {v.untyped_storage().data_ptr() for v in actual}.intersection(
                    v.untyped_storage().data_ptr() for v in repeated), name + ': live outputs alias')
                require(all(torch.equal(a, b) for a, b in zip(actual, snapshots)), name + ': outputs mutated')
                require(all(torch.equal(a, b) for a, b in zip(inputs, saved_inputs)), name + ': inputs mutated')
                if pattern == 'halfway':
                    require(actual[1][0, 0].item() == 1., 'Halfway rounding boundary was lost')
                elif pattern == 'zero':
                    require(all(torch.count_nonzero(v).item() == 0 for v in actual), 'Zero case failed')
                records.append({'name': name, 'shape': [m, h], 'dtype': label,
                                'max_abs_error_vs_fp64_oracle': error, 'launch_config': launch_config(h)})
                print(f'PASS {name}: max_abs_error={error:.8g}', flush=True)
    torch.cuda.synchronize()
    folder = ROOT / 'outputs/day9' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), ROOT / 'rmsnorm_lab/triton_kernel.py',
             ROOT / 'rmsnorm_lab/reference.py', ROOT / 'scripts/check_reference.py',
             ROOT / 'scripts/check_compiled_reference.py', contract_path]
    report = {
        'stage': 'first Triton kernel smoke checks; not full validation or performance',
        'cases': records, 'seed': 44, 'contract_sha256': sha(contract_path),
        'packages': {p: version(p) for p in ('torch', 'triton')},
        'gpu': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
        'file_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'prior_compiled_reference_checks': {str(p.relative_to(ROOT)): sha(p) for p in sorted(previous)},
        'initial_implementation_hidden_size_limit': 8192,
        'autotuned': False, 'performance_measured': False,
    }
    path = folder / 'triton_smoke_checks.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\nPASS: {len(records)} Triton smoke cases match the eager reference and FP64 oracle.')
    print('PASS: exact residual rounding, masked tails, input preservation, and fresh outputs.')
    print('Saved:', path)
    print('No speed conclusion yet. Full randomized/adversarial validation remains ahead.')


if __name__ == '__main__':
    main()
