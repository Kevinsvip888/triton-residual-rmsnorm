#!/usr/bin/env python3
"""Check the compiled baseline's correctness; do not benchmark its speed."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache/triton'))
sys.path.insert(0, str(ROOT))

import torch
from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
from check_reference import oracle


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_outputs(actual, expected, inputs, tolerances, label):
    require(isinstance(actual, tuple) and len(actual) == 2, label + ': expected two outputs')
    output, residual_out = actual
    x = inputs[0]
    for value in actual:
        require(value.shape == x.shape and value.dtype == x.dtype and value.device == x.device,
                label + ': wrong output shape, dtype, or device')
        require(value.is_contiguous() and torch.isfinite(value).all().item(),
                label + ': noncontiguous or nonfinite output')
    input_pointers = {v.untyped_storage().data_ptr() for v in inputs}
    output_pointers = [v.untyped_storage().data_ptr() for v in actual]
    require(len(set(output_pointers)) == 2 and not input_pointers.intersection(output_pointers),
            label + ': outputs must use independent fresh storage')
    expected_output, expected_residual = expected
    require(torch.equal(residual_out.cpu(), expected_residual.cpu()), label + ': residual rounding differs')
    torch.testing.assert_close(output.cpu().float(), expected_output.cpu().float(), **tolerances)
    return float((output.cpu().float() - expected_output.cpu().float()).abs().max().item())


def main():
    require(torch.cuda.is_available(), 'CUDA unavailable')
    contract_path = ROOT / 'specs/contract_v1.json'
    reference_path = ROOT / 'rmsnorm_lab/reference.py'
    oracle_path = ROOT / 'scripts/check_reference.py'
    contract = json.loads(contract_path.read_text())
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Unexpected contract')
    prior = []
    for path in (ROOT / 'outputs/day8').glob('*/reference_checks.json'):
        data = json.loads(path.read_text())
        if (data.get('contract_sha256') == sha(contract_path)
                and data.get('file_sha256', {}).get('rmsnorm_lab/reference.py') == sha(reference_path)
                and data.get('file_sha256', {}).get('scripts/check_reference.py') == sha(oracle_path)
                and len(data.get('valid_cases', [])) == 14
                and len(data.get('invalid_cases_rejected', [])) == 7):
            prior.append(path)
    require(prior, 'Run scripts/check_reference.py successfully with the current files first')
    config = {'backend': 'inductor', 'fullgraph': True, 'dynamic': False, 'mode': 'default'}
    compiled = torch.compile(residual_rmsnorm_reference, **config)
    torch.manual_seed(43)
    eps, rows = contract['eps'], []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for dtype, label, half_ulp in ((torch.float16, 'float16', 2**-11),
                                      (torch.bfloat16, 'bfloat16', 2**-8)):
            for pattern, m in (('random', 1), ('random', 32), ('halfway', 1), ('zero', 32)):
                h = 1024
                if pattern == 'random':
                    x = torch.randn(m, h, device='cuda', dtype=torch.float32).to(dtype)
                    residual = torch.randn(m, h, device='cuda', dtype=torch.float32).to(dtype)
                    weight = torch.randn(h, device='cuda', dtype=torch.float32).to(dtype)
                elif pattern == 'halfway':
                    x = torch.ones(m, h, device='cuda', dtype=dtype)
                    residual = torch.zeros_like(x)
                    residual[:, 0] = half_ulp
                    weight = torch.ones(h, device='cuda', dtype=dtype)
                else:
                    x = torch.zeros(m, h, device='cuda', dtype=dtype)
                    residual = torch.zeros_like(x)
                    weight = torch.ones(h, device='cuda', dtype=dtype)
                name = f'{label}/{pattern}/{m}x{h}'
                print('Checking ' + name + ' (first call for a new signature can compile)...', flush=True)
                validate_inputs(x, residual, weight, eps)
                inputs = (x, residual, weight)
                snapshots = [t.clone() for t in inputs]
                expected = oracle(x, residual, weight, eps)
                eager = residual_rmsnorm_reference(x, residual, weight, eps)
                tolerances = contract['output_tolerances'][label]
                check_outputs(eager, expected, inputs, tolerances, name + '/eager')
                result = compiled(x, residual, weight, eps)
                error = check_outputs(result, expected, inputs, tolerances, name + '/compiled')
                check_outputs(result, eager, inputs, tolerances, name + '/compiled-vs-eager')
                # A second call must allocate fresh outputs and preserve the first outputs.
                first_values = [v.clone() for v in result]
                repeated = compiled(x, residual, weight, eps)
                check_outputs(repeated, expected, inputs, tolerances, name + '/repeated')
                first_storage = {v.untyped_storage().data_ptr() for v in result}
                require(not first_storage.intersection(v.untyped_storage().data_ptr() for v in repeated),
                        name + ': repeated invocation reused live output storage')
                require(all(torch.equal(a, b) for a, b in zip(result, first_values)),
                        name + ': later invocation changed earlier outputs')
                require(all(torch.equal(a, b) for a, b in zip(inputs, snapshots)), name + ': input mutation')
                if pattern == 'halfway':
                    require(result[1][0, 0].item() == 1., name + ': rounding boundary not retained')
                rows.append({'name': name, 'shape': [m, h], 'dtype': label,
                             'max_abs_error_vs_fp64_oracle': error})
                print(f'PASS {name}: max_abs_error={error:.8g}', flush=True)
    torch.cuda.synchronize()
    directory = ROOT / 'outputs/day8' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        'stage': 'compiled reference correctness; no speed measurements',
        'compile_config': config, 'cases': rows, 'seed': 43,
        'packages': {p: version(p) for p in ('torch', 'triton')},
        'gpu': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
        'contract_sha256': sha(contract_path),
        'prior_reference_checks': {str(p.relative_to(ROOT)): sha(p) for p in sorted(prior)},
        'file_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__).resolve(), reference_path, oracle_path, contract_path)},
        'cache_directories': {k: os.environ[k] for k in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')},
        'fullgraph': 'Compilation is required without graph breaks; this is not a claim of one GPU kernel',
        'timings': None,
    }
    path = directory / 'compiled_reference_checks.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\nPASS: {len(rows)} compiled-baseline cases match eager PyTorch and the FP64 oracle.')
    print('PASS: residual rounding, inputs, fresh output storage, and repeated calls verified.')
    print('Saved:', path)
    print('No performance conclusion yet; compilation and test overhead are not benchmark latency.')


if __name__ == '__main__':
    main()
