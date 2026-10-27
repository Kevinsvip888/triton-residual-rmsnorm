#!/usr/bin/env python3
"""Validate the compiled baseline in isolated workers; no speed measurements."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache/triton'))
CONFIG = {'backend': 'inductor', 'fullgraph': True, 'dynamic': False, 'mode': 'default'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / 'rmsnorm_lab/reference.py',
             ROOT / 'scripts/check_reference.py', ROOT / 'scripts/check_compiled_reference.py',
             ROOT / 'scripts/validate_triton.py', ROOT / 'specs/contract_v1.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def gpu_identity():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'
    ], text=True).strip()
    require(len(text.splitlines()) == 1, 'This study expects one visible physical GPU')
    return text


def worker(args):
    # Import PyTorch only inside a worker. The controller never creates a CUDA context.
    sys.path.insert(0, str(ROOT))
    import torch
    from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
    from check_reference import oracle
    from check_compiled_reference import check_outputs
    from validate_triton import random_inputs

    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    contract = read(ROOT / 'specs/contract_v1.json')
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Unexpected contract')
    tolerances, eps = contract['output_tolerances'][args.dtype], contract['eps']
    compiled = torch.compile(residual_rmsnorm_reference, **CONFIG)
    rows, boundaries = [], []

    def check(inputs, name, seed):
        x, residual, weight = inputs
        validate_inputs(x, residual, weight, eps)
        before = [v.clone() for v in inputs]
        expected = oracle(x, residual, weight, eps)
        eager = residual_rmsnorm_reference(x, residual, weight, eps)
        actual = compiled(x, residual, weight, eps)
        error = check_outputs(actual, expected, inputs, tolerances, name + '/oracle')
        check_outputs(actual, eager, inputs, tolerances, name + '/eager')
        snapshots = [v.clone() for v in actual]
        second = compiled(x, residual, weight, eps)
        check_outputs(second, expected, inputs, tolerances, name + '/repeated')
        require(not {v.untyped_storage().data_ptr() for v in actual}.intersection(
            v.untyped_storage().data_ptr() for v in second), name + ': repeated outputs alias')
        require(all(torch.equal(a, b) for a, b in zip(actual, snapshots)), name + ': prior outputs changed')
        require(all(torch.equal(a, b) for a, b in zip(inputs, before)), name + ': inputs changed')
        actual_cpu, expected_cpu = actual[0].cpu().float(), expected[0].float()
        ratio = float(((actual_cpu - expected_cpu).abs() /
                      (tolerances['atol'] + tolerances['rtol'] * expected_cpu.abs())).max().item())
        require(ratio <= 1, name + ': tolerance exceeded')
        return {'name': name, 'dtype': args.dtype, 'shape': list(x.shape), 'seed': seed,
                'max_abs_error_vs_fp64_oracle': error, 'max_fraction_of_allowed_error': ratio,
                'residual_exact': True, 'inputs_preserved': True, 'fresh_outputs': True}

    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for m in contract['benchmark_main_shapes']['M']:
            print(f'Compiling/checking {args.dtype} {m}x{args.hidden_size}, seeds 0..9...', flush=True)
            for seed in range(10):
                name = f'{args.dtype}/random/{m}x{args.hidden_size}/seed{seed}'
                try:
                    rows.append(check(random_inputs(m, args.hidden_size, dtype, seed), name, seed))
                except Exception as error:
                    raise RuntimeError('Compiled validation failed: ' + name) from error
            print(f'PASS {args.dtype}, H={args.hidden_size}: {len(rows)}/30 random cases', flush=True)
        for pattern in ('halfway_up', 'halfway_down', 'zero', 'cancel'):
            m, h = (1 if pattern.startswith('halfway') else 32), args.hidden_size
            x, residual, weight = random_inputs(m, h, dtype, 1001)
            if pattern.startswith('halfway'):
                half = 2**-11 if dtype == torch.float16 else 2**-8
                x.fill_(1 if pattern == 'halfway_up' else 1 + 2 * half)
                residual.zero_(); weight.fill_(1)
                residual[:, 0] = half if pattern == 'halfway_up' else -half
            elif pattern == 'zero':
                x.zero_(); residual.zero_()
            else:
                residual = -x
            name = f'{args.dtype}/{pattern}/{m}x{h}'
            try:
                boundaries.append(check((x, residual, weight), name, 1001))
            except Exception as error:
                raise RuntimeError('Compiled validation failed: ' + name) from error
        torch.cuda.synchronize()
    require(len(rows) == 30 and len(boundaries) == 4, 'Worker case count mismatch')
    save(args.output, {
        'dtype': args.dtype, 'hidden_size': args.hidden_size,
        'main_cases': rows, 'boundary_cases': boundaries,
        'compile_config': CONFIG, 'contract_sha256': sha(ROOT / 'specs/contract_v1.json'),
        'file_sha256': source_hashes(), 'gpu_identity': gpu_identity(),
        'packages': {p: version(p) for p in ('torch', 'triton')},
        'cuda_build': torch.version.cuda, 'worker_pid': os.getpid(),
        'torch_cpu_threads_for_oracle': torch.get_num_threads(),
        'performance_measured': False,
    })
    print(f'PASS worker: {args.dtype} H={args.hidden_size}, 30 random + 4 boundary cases.', flush=True)


def controller():
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = read(contract_path)
    require(contract['benchmark_main_shapes'] == {'M': [1, 32, 2048], 'H': [768, 1024, 2560, 4096]}
            and contract['planned_random_seeds_per_shape_dtype'] == 10, 'Main matrix differs')
    prior = []
    for path in (ROOT / 'outputs/day10').glob('*/triton_validation.json'):
        report = read(path)
        if (report.get('contract_sha256') == sha(contract_path)
                and len(report.get('main_cases', [])) == 240
                and len(report.get('edge_cases', [])) == 30
                and len(report.get('invalid_inputs_rejected', [])) == 21
                and all(sha(ROOT / p) == digest for p, digest in report['file_sha256'].items())):
            prior.append(path)
    require(prior, 'A passing Triton full-matrix receipt for the current files is required')
    memcheck_log = ROOT / 'logs/day10_memcheck.log'
    memcheck_exit = ROOT / 'logs/day10_memcheck_exit_code.txt'
    require(memcheck_log.is_file() and memcheck_exit.is_file(), 'Saved memcheck log and exit code are required')
    memcheck_text = memcheck_log.read_text(errors='replace')
    summaries = re.findall(r'ERROR SUMMARY:\s*(\d+)\s+errors?', memcheck_text)
    require(summaries and all(int(n) == 0 for n in summaries)
            and memcheck_exit.read_text().strip() == '0'
            and 'PASS: 24 Triton smoke cases' in memcheck_text,
            'Memcheck did not record a clean, completed smoke run')
    memcheck_receipt = {str(p.relative_to(ROOT)): sha(p) for p in (memcheck_log, memcheck_exit)}
    expected_hashes, expected_gpu = source_hashes(), gpu_identity()
    folder = ROOT / 'outputs/day10' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    results, worker_files = [], []
    for dtype in ('float16', 'bfloat16'):
        for h in contract['benchmark_main_shapes']['H']:
            path = folder / f'compiled_{dtype}_h{h}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--dtype', dtype, '--hidden-size', str(h), '--output', str(path)]
            print(f'\nStarting isolated worker {len(results)+1}/8: {dtype}, H={h}', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            result = read(path)
            require(result['file_sha256'] == expected_hashes and result['gpu_identity'] == expected_gpu,
                    'Source or GPU changed during validation')
            require(result['compile_config'] == CONFIG and result['contract_sha256'] == sha(contract_path),
                    'Compile settings or contract changed')
            require(result['dtype'] == dtype and result['hidden_size'] == h
                    and len(result['main_cases']) == 30 and len(result['boundary_cases']) == 4,
                    'Worker result identity/count mismatch')
            if results:
                require(result['packages'] == results[0]['packages'], 'Worker packages differ')
            results.append(result); worker_files.append(path)
    random_rows = [row for result in results for row in result['main_cases']]
    edge_rows = [row for result in results for row in result['boundary_cases']]
    expected_pairs = {(d, m, h, s) for d in ('float16', 'bfloat16')
                      for m in (1, 32, 2048) for h in (768, 1024, 2560, 4096) for s in range(10)}
    actual_pairs = [(r['dtype'], *r['shape'], r['seed']) for r in random_rows]
    require(len(actual_pairs) == len(set(actual_pairs)) == 240 and set(actual_pairs) == expected_pairs,
            'Missing or duplicate main cases')
    require(len(edge_rows) == 32, 'Boundary case count mismatch')
    report = {
        'stage': 'compiled-baseline full benchmark-matrix correctness; no timing',
        'main_cases': random_rows, 'boundary_cases': edge_rows,
        'worker_count': len(results), 'compile_config': CONFIG,
        'packages': results[0]['packages'], 'gpu_identity': expected_gpu,
        'contract_sha256': sha(contract_path), 'file_sha256': expected_hashes,
        'worker_receipts_sha256': {str(p.relative_to(ROOT)): sha(p) for p in worker_files},
        'prior_triton_validation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(prior)},
        'saved_memcheck_evidence_sha256': memcheck_receipt,
        'max_fraction_of_allowed_error': max(r['max_fraction_of_allowed_error'] for r in random_rows + edge_rows),
        'performance_measured': False,
        'scope': 'Main compiled matrix plus four boundary patterns per dtype/hidden size; no arbitrary-shape correctness proof',
    }
    path = folder / 'compiled_matrix_validation.json'
    save(path, report)
    print('\nPASS: 240 compiled-baseline random cases and 32 boundary cases across 8 isolated workers.')
    print('PASS: fixed tolerances, exact residual values, input preservation, and independent outputs.')
    print('Maximum fraction of allowed error:', report['max_fraction_of_allowed_error'])
    print('Saved:', path)
    print('No performance conclusion; this run only checks correctness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), help=argparse.SUPPRESS)
    parser.add_argument('--hidden-size', type=int, choices=(768, 1024, 2560, 4096), help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        require(args.dtype is not None and args.hidden_size is not None and args.output is not None,
                'Internal worker arguments missing')
        worker(args)
    else:
        controller()


if __name__ == '__main__':
    main()
