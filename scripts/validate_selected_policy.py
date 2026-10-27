#!/usr/bin/env python3
"""Validate the frozen selected warp policy. This script never times or retunes it."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache/triton'))
sys.path.insert(0, str(ROOT))
import torch
from rmsnorm_lab.reference import residual_rmsnorm_reference
from rmsnorm_lab.tuned_kernel import (residual_rmsnorm_tuned, selected_launch_config,
                                     policy_sha256, POLICY_PATH)
from check_reference import oracle
from check_compiled_reference import check_outputs
from validate_triton import random_inputs, edge_inputs, EDGE_PATTERNS


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit_frozen_search():
    import tune_warps as tuning
    import benchmark_matrix as bm
    base = ROOT / 'outputs/day11/warp_tuning_v1'
    protocol_path = base / 'protocol.json'
    summary_path = base / 'tuning_summary.json'
    p, summary, policy = read(protocol_path), read(summary_path), read(POLICY_PATH)
    require(summary['protocol'] == p and summary['protocol_sha256'] == sha(protocol_path),
            'Tuning protocol differs')
    require(summary['selected_policy_sha256'] == sha(POLICY_PATH) == policy_sha256(),
            'Selected policy changed since search or process load')
    require(policy['tuning_protocol_sha256'] == sha(protocol_path), 'Policy protocol differs')
    require((summary['candidate_random_checks'], summary['candidate_boundary_checks'],
             summary['timing_batches']) == (1200, 480, 600), 'Search counts differ')
    require(bm.gpu_identity() == p['gpu_identity'], 'Different GPU from tuning')
    require(hashlib.sha256(bm.freeze_text().encode()).hexdigest() == p['environment_sha256'],
            'Environment changed since tuning')
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Search source changed: ' + relative)
    expected = set(policy['cases'])
    seen = set()
    require(len(summary['case_result_sha256']) == len(summary['cases']) == 24,
            'Expected 24 saved search results')
    for relative, digest in summary['case_result_sha256'].items():
        path = ROOT / relative
        require(sha(path) == digest, 'Search result changed: ' + relative)
        item = read(path)
        r = tuning.verify_result(path, p, item['dtype'], item['M'], item['H'])
        key = f"{r['dtype']}_m{r['M']}_h{r['H']}"
        require(key not in seen and key in expected, 'Duplicate/unexpected search case')
        require(policy['cases'][key] == {'dtype': r['dtype'], 'M': r['M'], 'H': r['H'],
                                        'num_warps': r['selected_warps']}, 'Selection differs')
        require(r in summary['cases'], 'Summary case differs from saved result')
        seen.add(key)
    require(seen == expected, 'Missing policy entries')
    sources = dict(p['source_sha256'])
    for relative in ('rmsnorm_lab/tuned_kernel.py', 'scripts/validate_selected_policy.py'):
        sources[relative] = sha(ROOT / relative)
    return sources, {
        'policy_sha256': sha(POLICY_PATH), 'tuning_protocol_sha256': sha(protocol_path),
        'tuning_summary_sha256': sha(summary_path), 'gpu_identity': p['gpu_identity'],
        'environment_sha256': p['environment_sha256'],
    }


def check_case(inputs, eps, contract, name, seed):
    x, residual, weight = inputs
    label = 'float16' if x.dtype == torch.float16 else 'bfloat16'
    tolerances = contract['output_tolerances'][label]
    before = [t.clone() for t in inputs]
    expected = oracle(x, residual, weight, eps)
    eager = residual_rmsnorm_reference(x, residual, weight, eps)
    check_outputs(eager, expected, inputs, tolerances, name + '/eager-vs-oracle')
    result = residual_rmsnorm_tuned(x, residual, weight, eps)
    error = check_outputs(result, expected, inputs, tolerances, name + '/triton-vs-oracle')
    check_outputs(result, eager, inputs, tolerances, name + '/triton-vs-eager')
    actual_cpu, expected_cpu = result[0].cpu().float(), expected[0].float()
    error_ratio = float(((actual_cpu - expected_cpu).abs() /
                        (tolerances['atol'] + tolerances['rtol'] * expected_cpu.abs())).max().item())
    require(error_ratio <= 1., name + ': elementwise tolerance exceeded')
    # Keep the first outputs alive while checking a second independent invocation.
    old_values = [t.clone() for t in result]
    repeated = residual_rmsnorm_tuned(x, residual, weight, eps)
    check_outputs(repeated, expected, inputs, tolerances, name + '/repeated')
    storage = {t.untyped_storage().data_ptr() for t in result}
    require(not storage.intersection(t.untyped_storage().data_ptr() for t in repeated),
            name + ': repeated call aliases a live output')
    require(all(torch.equal(a, b) for a, b in zip(result, old_values)), name + ': prior outputs changed')
    require(all(torch.equal(a, b) for a, b in zip(inputs, before)), name + ': inputs changed')
    return {'name': name, 'dtype': label, 'shape': list(x.shape), 'seed': seed, 'eps': eps,
            'max_abs_error_vs_fp64_oracle': error, 'max_fraction_of_allowed_error': error_ratio,
            'residual_exact': True, 'inputs_preserved': True, 'fresh_outputs': True}

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
                residual_rmsnorm_tuned(*arguments)
            except expected_error:
                rejected.append(name)
            else:
                raise RuntimeError('Invalid input accepted: ' + name)
    return rejected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='One fresh random seed for every policy entry plus all 30 edges; suitable for memcheck')
    args = parser.parse_args()
    source_hashes, evidence = audit_frozen_search()
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = read(contract_path)
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Contract changed')
    require(contract['output_tolerances'] == {
        'float16': {'rtol': .002, 'atol': .0002},
        'bfloat16': {'rtol': .01, 'atol': .001}}, 'Frozen tolerances changed')
    if args.smoke:
        prior = []
        for path in (ROOT / 'outputs/day11/selected_validation_v1').glob('*/full_validation.json'):
            r = read(path)
            if (r.get('file_sha256') == source_hashes and r.get('evidence') == evidence
                    and len(r.get('main_cases', [])) == 240 and len(r.get('edge_cases', [])) == 30
                    and len(r.get('invalid_inputs_rejected', [])) == 21
                    and path.with_suffix('.sha256').read_text().strip() == sha(path)):
                prior.append(path)
        require(prior, 'Run this script without --smoke successfully first')
    else:
        prior = []
    records, edges, dispatched = [], [], {}
    seeds = [2027] if args.smoke else list(range(10))
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for dtype, label in ((torch.float16, 'float16'), (torch.bfloat16, 'bfloat16')):
            for m in (1, 32, 2048):
                for h in (768, 1024, 2560, 4096):
                    config = selected_launch_config(dtype, m, h)
                    require(config['matched_policy'], 'Main shape missed selected policy')
                    key = f'{label}_m{m}_h{h}'
                    dispatched[key] = config
                    print(f"Checking {label} {m}x{h}: selected {config['num_warps']} warps...", flush=True)
                    for seed in seeds:
                        r = check_case(random_inputs(m, h, dtype, seed), contract['eps'], contract,
                                       f'{label}/random/{m}x{h}/seed{seed}', seed)
                        r['launch_config'] = config
                        records.append(r)
                    print(f'PASS {key}: {len(seeds)} random checks', flush=True)
            for pattern in EDGE_PATTERNS:
                inputs, eps = edge_inputs(pattern, dtype)
                config = selected_launch_config(dtype, *inputs[0].shape)
                r = check_case(inputs, eps, contract, f'{label}/edge/{pattern}', 1001)
                r['launch_config'] = config
                edges.append(r)
                print(f"PASS {label}/edge/{pattern}: {config['num_warps']} warps, "
                      f"{'selected' if config['matched_policy'] else 'fallback'}", flush=True)
    rejected = invalid_cases()
    expected_count = 24 if args.smoke else 240
    require(len(records) == expected_count and len(edges) == 30 and len(rejected) == 21,
            'Validation counts differ')
    require(len(dispatched) == 24, 'All selected entries must execute')
    torch.cuda.synchronize()
    for relative, digest in source_hashes.items():
        require(sha(ROOT / relative) == digest, 'Source changed during validation: ' + relative)
    require(sha(POLICY_PATH) == evidence['policy_sha256'], 'Policy changed during validation')
    folder = ROOT / 'outputs/day11/selected_validation_v1' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        'stage': 'Frozen selected-policy validation; no tuning or performance measurements',
        'mode': 'smoke' if args.smoke else 'full', 'evidence': evidence,
        'file_sha256': source_hashes, 'contract_sha256': sha(contract_path),
        'main_cases': records, 'edge_cases': edges, 'invalid_inputs_rejected': rejected,
        'executed_policy_entries': dispatched, 'random_seeds': seeds, 'edge_seed': 1001,
        'prior_full_validation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in prior},
        'packages': {p: version(p) for p in ('torch', 'triton')},
        'gpu': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
        'max_fraction_of_allowed_error': max(r['max_fraction_of_allowed_error'] for r in records + edges),
        'allocator_debug_setting': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', ''),
        'performance_measured': False,
        'scope': 'Finite moderate-range inputs. External Compute Sanitizer output and exit code establish memory-check status.',
    }
    path = folder / ('smoke_validation.json' if args.smoke else 'full_validation.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    path.with_suffix('.sha256').write_text(sha(path) + '\n')
    print(f"\nPASS: {expected_count} selected-policy random cases, 30 edge cases, 21 invalid-input checks.")
    print('PASS: all 24 policy entries executed; exact residual rounding, fixed tolerances, input preservation and fresh outputs.')
    print('Maximum fraction of allowed error:', report['max_fraction_of_allowed_error'])
    print('Saved:', path)
    print('No performance conclusion. Frozen search files and original kernel were not modified.')


if __name__ == '__main__':
    main()
