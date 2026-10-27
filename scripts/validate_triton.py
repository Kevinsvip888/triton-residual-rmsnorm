#!/usr/bin/env python3
"""Day 10: fixed-matrix and edge-case validation, without timing or tuning."""
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
from rmsnorm_lab.triton_kernel import residual_rmsnorm_triton
from check_reference import oracle
from check_compiled_reference import check_outputs


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_inputs(m, h, dtype, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    def sample(shape):
        return torch.randn(shape, device='cuda', generator=generator, dtype=torch.float32).to(dtype)
    return sample((m, h)), sample((m, h)), sample((h,))


def check_case(inputs, eps, contract, name, seed):
    x, residual, weight = inputs
    label = 'float16' if x.dtype == torch.float16 else 'bfloat16'
    tolerances = contract['output_tolerances'][label]
    before = [t.clone() for t in inputs]
    expected = oracle(x, residual, weight, eps)
    eager = residual_rmsnorm_reference(x, residual, weight, eps)
    check_outputs(eager, expected, inputs, tolerances, name + '/eager-vs-oracle')
    result = residual_rmsnorm_triton(x, residual, weight, eps)
    error = check_outputs(result, expected, inputs, tolerances, name + '/triton-vs-oracle')
    check_outputs(result, eager, inputs, tolerances, name + '/triton-vs-eager')
    actual_cpu, expected_cpu = result[0].cpu().float(), expected[0].float()
    error_ratio = float(((actual_cpu - expected_cpu).abs() /
                        (tolerances['atol'] + tolerances['rtol'] * expected_cpu.abs())).max().item())
    require(error_ratio <= 1., name + ': elementwise tolerance exceeded')
    # Keep the first outputs alive while checking a second independent invocation.
    old_values = [t.clone() for t in result]
    repeated = residual_rmsnorm_triton(x, residual, weight, eps)
    check_outputs(repeated, expected, inputs, tolerances, name + '/repeated')
    storage = {t.untyped_storage().data_ptr() for t in result}
    require(not storage.intersection(t.untyped_storage().data_ptr() for t in repeated),
            name + ': repeated call aliases a live output')
    require(all(torch.equal(a, b) for a, b in zip(result, old_values)), name + ': prior outputs changed')
    require(all(torch.equal(a, b) for a, b in zip(inputs, before)), name + ': inputs changed')
    return {'name': name, 'dtype': label, 'shape': list(x.shape), 'seed': seed, 'eps': eps,
            'max_abs_error_vs_fp64_oracle': error, 'max_fraction_of_allowed_error': error_ratio,
            'residual_exact': True, 'inputs_preserved': True, 'fresh_outputs': True}


EDGE_PATTERNS = ('zeros', 'cancel', 'almost_cancel', 'small', 'large',
                 'single_outlier', 'alternating', 'gamma_zero', 'gamma_negative',
                 'halfway_up', 'halfway_down', 'odd_tail', 'upper_limit',
                 'single_value', 'eps_override')


def edge_inputs(pattern, dtype):
    shapes = {'odd_tail': (2, 4097), 'upper_limit': (2, 8192),
              'single_value': (1, 1), 'halfway_up': (1, 1024), 'halfway_down': (1, 1024),
              'eps_override': (4, 768)}
    m, h = shapes.get(pattern, (3, 257))
    x, residual, weight = random_inputs(m, h, dtype, 1001)
    eps = 1e-6
    if pattern == 'zeros':
        x.zero_(); residual.zero_()
    elif pattern == 'cancel':
        residual = -x
    elif pattern == 'almost_cancel':
        x.fill_(128); residual.fill_(-127.5); weight.fill_(1)
    elif pattern == 'small':
        x.fill_(1e-3); residual.zero_(); weight.fill_(1)
    elif pattern == 'large':
        x.fill_(256); residual.fill_(128)
    elif pattern in ('single_outlier', 'odd_tail'):
        x.zero_(); residual.zero_(); weight.fill_(1)
        x[:, -1] = 4.; residual[:, -1] = 2.
    elif pattern == 'alternating':
        signs = torch.where(torch.arange(h, device='cuda') % 2 == 0, 1., -1.).to(dtype)
        x[:] = signs; residual[:] = -.5 * signs; weight.fill_(1)
    elif pattern == 'gamma_zero':
        weight.zero_()
    elif pattern == 'gamma_negative':
        weight.fill_(-2)
    elif pattern in ('halfway_up', 'halfway_down'):
        half = 2**-11 if dtype == torch.float16 else 2**-8
        x.fill_(1 if pattern == 'halfway_up' else 1 + 2 * half)
        residual.zero_(); weight.fill_(1)
        residual[:, 0] = half if pattern == 'halfway_up' else -half
    elif pattern == 'single_value':
        x.fill_(1); residual.fill_(2); weight.fill_(-3)
    elif pattern == 'eps_override':
        eps = 1e-3
    return (x, residual, weight), eps


def invalid_cases():
    x = torch.ones(2, 4, device='cuda', dtype=torch.float16)
    residual = x.clone()
    weight = torch.ones(4, device='cuda', dtype=x.dtype)
    cases = [
        ('non-tensor input', (None, residual, weight, 1e-6), TypeError),
        ('rank-one x', (x[0], residual[0], weight, 1e-6), ValueError),
        ('empty M', (x[:0], residual[:0], weight, 1e-6), ValueError),
        ('empty H', (x[:, :0], residual[:, :0], weight[:0], 1e-6), ValueError),
        ('wrong residual shape', (x, residual[:1], weight, 1e-6), ValueError),
        ('wrong gamma shape', (x, residual, weight[:3], 1e-6), ValueError),
        ('rank-two gamma', (x, residual, weight[None, :], 1e-6), ValueError),
        ('FP32 inputs', (x.float(), residual.float(), weight.float(), 1e-6), TypeError),
        ('mixed dtype', (x, residual.bfloat16(), weight, 1e-6), TypeError),
        ('CPU tensors', (x.cpu(), residual.cpu(), weight.cpu(), 1e-6), ValueError),
        ('mixed CPU CUDA', (x, residual.cpu(), weight, 1e-6), ValueError),
        ('strided x', (x.t(), residual.t().contiguous(), weight[:2], 1e-6), ValueError),
        ('strided gamma', (x, residual, torch.ones(8, device='cuda', dtype=x.dtype)[::2], 1e-6), ValueError),
        ('zero eps', (x, residual, weight, 0.), ValueError),
        ('negative eps', (x, residual, weight, -1.), ValueError),
        ('infinite eps', (x, residual, weight, float('inf')), ValueError),
        ('nan eps', (x, residual, weight, float('nan')), ValueError),
        ('boolean eps', (x, residual, weight, True), TypeError),
        ('string eps', (x, residual, weight, '1e-6'), TypeError),
    ]
    large = torch.ones(1, 8193, device='cuda', dtype=x.dtype)
    cases.append(('initial H limit', (large, large.clone(), large[0].clone(), 1e-6), ValueError))
    grad_x = torch.ones(2, 4, device='cuda', dtype=x.dtype, requires_grad=True)
    cases.append(('autograd request', (grad_x, residual, weight, 1e-6), RuntimeError))
    rejected = []
    with torch.enable_grad():
        for name, arguments, expected_error in cases:
            try:
                residual_rmsnorm_triton(*arguments)
            except expected_error:
                rejected.append(name)
            else:
                raise RuntimeError('Invalid input accepted: ' + name)
    return rejected


def main():
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = json.loads(contract_path.read_text())
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Unexpected contract')
    require(contract['benchmark_main_shapes'] == {'M': [1, 32, 2048], 'H': [768, 1024, 2560, 4096]}
            and contract['planned_random_seeds_per_shape_dtype'] == 10, 'Main matrix changed')
    prior = []
    for path in (ROOT / 'outputs/day9').glob('*/triton_smoke_checks.json'):
        receipt = json.loads(path.read_text())
        if (receipt.get('contract_sha256') == sha(contract_path) and len(receipt.get('cases', [])) == 24
                and all(sha(ROOT / p) == digest for p, digest in receipt['file_sha256'].items())):
            prior.append(path)
    require(prior, 'A passing smoke receipt for the unchanged kernel/contract is required')
    records, edges = [], []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for dtype, label in ((torch.float16, 'float16'), (torch.bfloat16, 'bfloat16')):
            for m in contract['benchmark_main_shapes']['M']:
                for h in contract['benchmark_main_shapes']['H']:
                    print(f'Validating {label} {m}x{h}: seeds 0..9...', flush=True)
                    for seed in range(10):
                        name = f'{label}/random/{m}x{h}/seed{seed}'
                        try:
                            records.append(check_case(random_inputs(m, h, dtype, seed), contract['eps'],
                                                      contract, name, seed))
                        except Exception as error:
                            raise RuntimeError('Validation failed: ' + name) from error
                    worst = max(r['max_fraction_of_allowed_error'] for r in records[-10:])
                    print(f'PASS {len(records)}/240 random cases; worst allowed-error fraction={worst:.6f}', flush=True)
            for pattern in EDGE_PATTERNS:
                inputs, eps = edge_inputs(pattern, dtype)
                name = f'{label}/edge/{pattern}'
                try:
                    edges.append(check_case(inputs, eps, contract, name, 1001))
                except Exception as error:
                    raise RuntimeError('Validation failed: ' + name) from error
                print(f'PASS {label}/edge/{pattern}', flush=True)
    rejected = invalid_cases()
    require(len(records) == 240 and len(edges) == 30 and len(rejected) == 21, 'Validation count mismatch')
    torch.cuda.synchronize()
    folder = ROOT / 'outputs/day10' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), ROOT / 'rmsnorm_lab/triton_kernel.py',
             ROOT / 'rmsnorm_lab/reference.py', ROOT / 'scripts/check_reference.py',
             ROOT / 'scripts/check_compiled_reference.py', ROOT / 'scripts/check_triton.py', contract_path]
    report = {
        'stage': 'Day 10 Triton numerical and API validation; no performance measurements',
        'main_cases': records, 'edge_cases': edges, 'invalid_inputs_rejected': rejected,
        'random_seeds': list(range(10)), 'edge_seed': 1001,
        'contract_sha256': sha(contract_path),
        'file_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'prior_smoke_receipts': {str(p.relative_to(ROOT)): sha(p) for p in sorted(prior)},
        'packages': {p: version(p) for p in ('torch', 'triton')},
        'gpu': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
        'torch_cpu_threads_for_oracle': torch.get_num_threads(),
        'max_fraction_of_allowed_error': max(r['max_fraction_of_allowed_error'] for r in records + edges),
        'sanitizer_run': False, 'full_compiled_matrix_checked': False, 'performance_measured': False,
        'scope': 'Finite moderate-range numerical tests and wrapper rejection checks; not proof for every possible input',
    }
    path = folder / 'triton_validation.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\nPASS: {len(records)} random cases, {len(edges)} edge cases, {len(rejected)} invalid-input checks.')
    print('PASS: frozen tolerances, exact residual values, input preservation, and independent outputs.')
    print('Maximum fraction of allowed error:', report['max_fraction_of_allowed_error'])
    print('Saved:', path)
    print('No performance conclusion. GPU memory-safety tooling and full compiled-matrix validation remain separate checks.')


if __name__ == '__main__':
    main()
