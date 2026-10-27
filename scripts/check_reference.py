#!/usr/bin/env python3
"""Day 8 GPU reference checks against hand cases and a CPU FP64 oracle."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(x, residual, weight, eps):
    """CPU FP64 normalization, after the specified rounding of the residual sum."""
    xd = x.detach().cpu().double()
    rd = residual.detach().cpu().double()
    gd = weight.detach().cpu().double()
    rounded = (xd + rd).to(x.dtype)
    values = rounded.double()
    denominator = ((values * values).sum(dim=-1, keepdim=True) / x.shape[-1] + eps).sqrt()
    output = (values / denominator * gd).to(x.dtype)
    return output, rounded


def evaluate_case(x, residual, weight, eps, tolerances, name):
    validate_inputs(x, residual, weight, eps)
    snapshots = [t.clone() for t in (x, residual, weight)]
    output, rounded = residual_rmsnorm_reference(x, residual, weight, eps)
    expected, expected_rounded = oracle(x, residual, weight, eps)
    for actual in (output, rounded):
        require(actual.shape == x.shape and actual.dtype == x.dtype and actual.device == x.device,
                name + ': output metadata differs')
        require(actual.is_contiguous(), name + ': output is not contiguous')
        require(torch.isfinite(actual).all().item(), name + ': output is not finite')
    storage = [t.untyped_storage().data_ptr() for t in (x, residual, weight, output, rounded)]
    require(storage[-2] not in storage[:3] and storage[-1] not in storage[:3]
            and storage[-1] != storage[-2], name + ': outputs must have independent fresh storage')
    for original, snapshot in zip((x, residual, weight), snapshots):
        require(torch.equal(original, snapshot), name + ': input mutated')
    require(torch.equal(rounded.cpu(), expected_rounded), name + ': residual rounding differs from oracle')
    actual_cpu = output.cpu().float()
    expected_cpu = expected.float()
    torch.testing.assert_close(actual_cpu, expected_cpu, **tolerances)
    error = float((actual_cpu - expected_cpu).abs().max().item())
    print(f'PASS {name}: shape={tuple(x.shape)}, max_abs_error={error:.8g}', flush=True)
    return {'name': name, 'shape': list(x.shape), 'dtype': str(x.dtype),
            'max_absolute_error_vs_rounded_fp64_oracle': error, 'eps': eps}


def main():
    require(torch.cuda.is_available(), 'CUDA unavailable; run in the existing serving environment')
    require(torch.cuda.is_bf16_supported(), 'BF16 unavailable')
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = json.loads(contract_path.read_text())
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Unexpected contract')
    eps = contract['eps']
    torch.manual_seed(42)
    cases, rejected = [], []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for dtype, label, half_ulp in ((torch.float16, 'float16', 2**-11),
                                       (torch.bfloat16, 'bfloat16', 2**-8)):
            tolerances = contract['output_tolerances'][label]
            for m, h in ((1, 4), (1, 768), (32, 1024), (17, 257)):
                x = torch.randn(m, h, device='cuda', dtype=torch.float32).to(dtype)
                residual = torch.randn(m, h, device='cuda', dtype=torch.float32).to(dtype)
                weight = torch.randn(h, device='cuda', dtype=torch.float32).to(dtype)
                cases.append(evaluate_case(x, residual, weight, eps, tolerances,
                                           f'{label}/random/{m}x{h}'))

            # Hand calculation: rounded sum [2,0,0,0], mean square=1.
            x = torch.tensor([[1., 0., 0., 0.]], device='cuda', dtype=dtype)
            residual = x.clone()
            weight = torch.tensor([1., 2., -1., .5], device='cuda', dtype=dtype)
            cases.append(evaluate_case(x, residual, weight, eps, tolerances, label + '/hand'))
            actual, rounded = residual_rmsnorm_reference(x, residual, weight, eps)
            hand = torch.tensor([[2 / (1 + eps)**.5, 0., 0., 0.]], device='cuda', dtype=dtype)
            torch.testing.assert_close(actual.float(), hand.float(), **tolerances)
            require(torch.equal(rounded, torch.tensor([[2., 0., 0., 0.]], device='cuda', dtype=dtype)),
                    'Hand residual sum failed')

            # Zero row must stay finite and produce exactly zero.
            zero = torch.zeros(2, 257, device='cuda', dtype=dtype)
            weight = torch.ones(257, device='cuda', dtype=dtype)
            cases.append(evaluate_case(zero, zero.clone(), weight, eps, tolerances, label + '/zero'))
            output, rounded = residual_rmsnorm_reference(zero, zero, weight, eps)
            require(torch.count_nonzero(output).item() == 0 and torch.count_nonzero(rounded).item() == 0,
                    'Zero row failed')

            # Halfway residual addition distinguishes this contract from FP32-only fusion.
            x = torch.ones(1, 4, device='cuda', dtype=dtype)
            residual = torch.zeros_like(x)
            residual[0, 0] = half_ulp
            weight = torch.ones(4, device='cuda', dtype=dtype)
            cases.append(evaluate_case(x, residual, weight, eps, tolerances, label + '/halfway'))
            _, rounded = residual_rmsnorm_reference(x, residual, weight, eps)
            require(rounded[0, 0].item() == 1., 'Expected round-to-nearest-even at halfway point')
            require((x.float() + residual.float())[0, 0].item() > 1., 'Halfway test has no rounding boundary')

        x = torch.ones(2, 4, device='cuda', dtype=torch.float16)
        residual, weight = x.clone(), torch.ones(4, device='cuda', dtype=x.dtype)
        invalid = [
            ('wrong residual shape', (x, residual[:, :3], weight, eps)),
            ('wrong gamma shape', (x, residual, weight[:3], eps)),
            ('mixed dtypes', (x, residual.bfloat16(), weight, eps)),
            ('noncontiguous input', (x.t(), residual.t(), torch.ones(2, device='cuda', dtype=x.dtype), eps)),
            ('CPU input', (x.cpu(), residual.cpu(), weight.cpu(), eps)),
            ('nonpositive eps', (x, residual, weight, 0.)),
            ('nonfinite eps', (x, residual, weight, float('nan'))),
        ]
        for name, arguments in invalid:
            try:
                validate_inputs(*arguments)
            except (TypeError, ValueError):
                rejected.append(name)
            else:
                raise RuntimeError('Invalid input accepted: ' + name)
    torch.cuda.synchronize()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'outputs/day8' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        'stage': 'Day 8 eager reference validation; no Triton kernel or performance result yet',
        'packages': {p: version(p) for p in ('torch', 'triton')},
        'gpu': torch.cuda.get_device_name(0),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'cuda_build': torch.version.cuda,
        'contract_sha256': sha(contract_path),
        'file_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / 'rmsnorm_lab/reference.py', Path(__file__).resolve(), contract_path)},
        'oracle': 'CPU float64; residual sum explicitly rounded to input dtype before normalization',
        'seed': 42, 'valid_cases': cases, 'invalid_cases_rejected': rejected,
        'input_mutation': False,
    }
    (directory / 'reference_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'\nPASS: {len(cases)} reference cases and {len(rejected)} invalid-input checks.')
    print('PASS: FP16/BF16 rounding, hand calculation, zero rows, input preservation, and output storage.')
    print('Saved:', directory / 'reference_checks.json')
    print('No Triton performance measurements have been made.')


if __name__ == '__main__':
    main()
