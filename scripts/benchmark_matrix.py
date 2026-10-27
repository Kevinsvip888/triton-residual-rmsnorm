#!/usr/bin/env python3
"""Measure the initial kernel across 24 cases with the pilot's graph timing scope."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/day11/baseline_v1'
METHODS = ('eager', 'compiled', 'triton')
DTYPES, ROWS, HIDDEN = ('float16', 'bfloat16'), (1, 32, 2048), (768, 1024, 2560, 4096)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def gpu_identity():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader,nounits'
    ], text=True).strip()
    require(len(text.splitlines()) == 1, 'Expected one physical GPU')
    fields = [v.strip() for v in text.split(',')]
    require(len(fields) == 3, 'Unexpected GPU identity output')
    return dict(zip(('uuid', 'name', 'driver'), fields))


def freeze_text():
    return subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable],
                                    text=True, stderr=subprocess.PIPE)


def protocol():
    pilot_path = ROOT / 'outputs/day11/pilot_protocol_v1.json'
    pilot = read(pilot_path)
    require(pilot['shape'] == [32, 4096] and pilot['dtype'] == 'bfloat16'
            and pilot['captured_calls_per_replay'] == 64 and pilot['replays_per_batch'] == 200
            and pilot['repetitions_per_method'] == 6, 'Pilot timing scope differs')
    for p, digest in pilot['source_sha256'].items():
        require(sha(ROOT / p) == digest, 'Initial source changed since the pilot: ' + p)
    identity = gpu_identity()
    require(identity == pilot['gpu_identity'], 'GPU differs from the pilot; keep studies separate')
    environment_hash = hashlib.sha256(freeze_text().encode()).hexdigest()
    require(environment_hash == pilot['environment_sha256'], 'Environment differs from the pilot')
    sources = {**pilot['source_sha256'], str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())}
    return {
        'study': 'day11_initial_kernel_matrix_v1', 'pilot_protocol_sha256': sha(pilot_path),
        'dtypes': list(DTYPES), 'M': list(ROWS), 'H': list(HIDDEN), 'seed': 123,
        'primary_case': {'dtype': 'bfloat16', 'M': 32, 'H': 4096},
        'compile_config': pilot['compile_config'], 'eps': pilot['eps'],
        'captured_calls_per_replay': 64, 'replays_per_batch': 200, 'repetitions_per_method': 6,
        'method_orders': pilot['method_orders'], 'warmup_calls_per_method': 10,
        'warmup_replays_per_method': 5, 'mixed_graph_warmup_seconds': 1.,
        'metric': pilot['metric'], 'aggregation': pilot['aggregation'],
        'outputs': pilot['outputs'], 'cache_scope': pilot['cache_scope'], 'excluded': pilot['excluded'],
        'gpu_identity': identity, 'packages': pilot['packages'], 'cuda_build': pilot['cuda_build'],
        'environment_sha256': environment_hash, 'source_sha256': sources,
        'worker_policy': 'one fresh process per shape/dtype, sequential; six within-process timing batches',
        'scope': 'warm CUDA Graph replay, before any kernel tuning; not Python-call or model latency',
    }


def worker(args):
    sys.path.insert(0, str(ROOT))
    import torch
    import benchmark_pilot as bp
    from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
    from rmsnorm_lab.triton_kernel import residual_rmsnorm_triton_unchecked, launch_config
    from check_reference import oracle
    from check_compiled_reference import check_outputs
    from validate_triton import random_inputs

    bp.require_exclusive_gpu()
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    setup_pids = bp.active_gpu_pids()
    require(len(setup_pids) <= 1, 'Multiple compute processes appeared during setup')
    allowed_pids = setup_pids | {os.getpid()}
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    for name in ('CUDA_LAUNCH_BLOCKING', 'PYTORCH_NO_CUDA_MEMORY_CACHING',
                 'TRITON_INTERPRET', 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE'):
        require(os.environ.get(name, '') in ('', '0'), 'Debug setting active during timing: ' + name)
    p = read(BASE / 'protocol.json')
    require(bp.COMPILE_CONFIG == p['compile_config'] and bp.CAPTURED_CALLS == 64
            and bp.REPLAYS_PER_BATCH == 200 and bp.REPETITIONS == 6, 'Pilot helper settings changed')
    require(gpu_identity() == p['gpu_identity'], 'Worker GPU differs')
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Worker source differs: ' + relative)
    freeze = freeze_text()
    require(hashlib.sha256(freeze.encode()).hexdigest() == p['environment_sha256'], 'Worker environment differs')
    contract = read(ROOT / 'specs/contract_v1.json')
    require(contract['eps'] == p['eps'], 'Contract epsilon changed')
    evidence = {}
    evidence.update(bp.checked_receipts('outputs/day10/*/triton_validation.json',
                    {'main_cases': 240, 'edge_cases': 30, 'invalid_inputs_rejected': 21}))
    compiled_receipts = bp.checked_receipts('outputs/day10/*/compiled_matrix_validation.json',
                                          {'main_cases': 240, 'boundary_cases': 32})
    evidence.update(compiled_receipts)
    for relative in compiled_receipts:
        report = read(ROOT / relative)
        require(report['compile_config'] == p['compile_config'], 'Compiled validation settings differ')
        for path, digest in report['saved_memcheck_evidence_sha256'].items():
            require(sha(ROOT / path) == digest, 'Memcheck evidence changed')
            evidence[path] = digest
    folder = args.output.parent
    (folder / 'environment.txt').write_text(freeze)
    before = bp.gpu_snapshot()
    save(folder / 'gpu_before.json', before)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    samples = {method: [] for method in METHODS}
    graphs, outputs = {}, {}
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        inputs = random_inputs(args.rows, args.hidden_size, dtype, p['seed'])
        validate_inputs(*inputs, p['eps'])
        before_inputs = [v.clone() for v in inputs]
        expected = oracle(*inputs, p['eps'])
        methods = {'eager': residual_rmsnorm_reference,
                   'compiled': torch.compile(residual_rmsnorm_reference, **p['compile_config']),
                   'triton': residual_rmsnorm_triton_unchecked}
        for method in METHODS:
            print('  Preparing/capturing ' + method + ' (outside timing)...', flush=True)
            graphs[method], outputs[method] = bp.capture(methods[method], inputs, p['eps'])
            graphs[method].replay()
            torch.cuda.synchronize()
            pointers = [v.untyped_storage().data_ptr() for pair in outputs[method] for v in pair]
            require(len(set(pointers)) == 128, 'Captured invocations alias live outputs')
            for pair in outputs[method]:
                check_outputs(pair, expected, inputs, contract['output_tolerances'][args.dtype], method + '/capture')
        for method in METHODS:
            for _ in range(5):
                graphs[method].replay()
        torch.cuda.synchronize()
        deadline = time.perf_counter() + 1.
        while time.perf_counter() < deadline:
            for method in METHODS:
                for _ in range(10):
                    graphs[method].replay()
            torch.cuda.synchronize()
        bp.require_exclusive_gpu(allowed_pids)
        for repetition, order in enumerate(p['method_orders'], 1):
            measured = {}
            for method in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(200):
                    graphs[method].replay()
                end.record(); end.synchronize()
                elapsed_ms = start.elapsed_time(end)
                us = elapsed_ms * 1000 / (64 * 200)
                require(math.isfinite(us) and us > 0, 'Invalid CUDA-event measurement')
                samples[method].append({'repetition': repetition, 'batch_elapsed_ms': elapsed_ms,
                                        'mean_us_per_invocation': us})
                measured[method] = us
            print('  Rep ' + str(repetition) + '/6: ' + ', '.join(
                f'{method}={measured[method]:.4f} us' for method in METHODS), flush=True)
        for method in METHODS:
            for pair in outputs[method]:
                check_outputs(pair, expected, inputs, contract['output_tolerances'][args.dtype], method + '/post-timing')
        require(all(torch.equal(a, b) for a, b in zip(inputs, before_inputs)), 'Inputs mutated during timing')
    bp.require_exclusive_gpu(allowed_pids)
    after = bp.gpu_snapshot()
    require({key: after[key] for key in ('uuid', 'name', 'driver')} == p['gpu_identity'], 'GPU changed')
    for relative, digest in {**p['source_sha256'], **evidence}.items():
        require(sha(ROOT / relative) == digest, 'Source/evidence changed during timing: ' + relative)
    summaries, speedups = bp.summarize_samples(samples)
    save(folder / 'gpu_after.json', after)
    result = {
        'dtype': args.dtype, 'M': args.rows, 'H': args.hidden_size,
        'protocol_sha256': sha(BASE / 'protocol.json'), 'source_sha256': p['source_sha256'],
        'gpu_identity': p['gpu_identity'], 'environment_sha256': p['environment_sha256'],
        'triton_launch_config': launch_config(args.hidden_size), 'seed': p['seed'],
        'summaries': summaries, 'speedups_ratio_of_medians': speedups, 'batches': samples,
        'validation_and_memcheck_receipts': evidence,
        'gpu_before': before, 'gpu_after': after, 'gpu_compute_pids_at_setup': sorted(setup_pids),
        'run_file_sha256': {str(path.relative_to(ROOT)): sha(path) for path in (
            folder / 'environment.txt', folder / 'gpu_before.json', folder / 'gpu_after.json')},
        'note': 'Six descriptive timing batches in one process; static-input GPU replay only',
    }
    save(args.output, result)
    args.output.with_suffix('.sha256').write_text(sha(args.output) + '\n')
    print(f'PASS case: {args.dtype} [{args.rows},{args.hidden_size}]; '
          f'Triton/compiled speedup={speedups["compiled_over_triton"]:.4f}x', flush=True)


def verify_result(path, p, dtype, m, h):
    require(path.with_suffix('.sha256').read_text().strip() == sha(path), 'Case result hash differs')
    result = read(path)
    require((result['dtype'], result['M'], result['H']) == (dtype, m, h), 'Case identity differs')
    require(result['protocol_sha256'] == sha(BASE / 'protocol.json')
            and result['source_sha256'] == p['source_sha256']
            and result['gpu_identity'] == p['gpu_identity']
            and result['environment_sha256'] == p['environment_sha256'], 'Case protocol differs')
    for relative, digest in {**result['run_file_sha256'], **result['validation_and_memcheck_receipts']}.items():
        require(sha(ROOT / relative) == digest, 'Case evidence changed: ' + relative)
    import statistics
    for method in METHODS:
        batches = result['batches'][method]
        require(len(batches) == 6 and {b['repetition'] for b in batches} == set(range(1, 7)), 'Missing timing batches')
        values = []
        for batch in batches:
            value = batch['mean_us_per_invocation']
            require(math.isfinite(value) and value > 0 and math.isclose(
                value, batch['batch_elapsed_ms'] * 1000 / (64 * 200), rel_tol=1e-12), 'Timing units/counts differ')
            values.append(value)
        require(math.isclose(statistics.median(values), result['summaries'][method]['median_us'], rel_tol=1e-12), 'Median differs')
    for baseline in ('eager', 'compiled'):
        ratio = result['summaries'][baseline]['median_us'] / result['summaries']['triton']['median_us']
        require(math.isclose(ratio, result['speedups_ratio_of_medians'][baseline + '_over_triton'], rel_tol=1e-12), 'Speedup differs')
    return result


def controller():
    p = protocol()
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / 'protocol.json'
    if path.exists():
        require(read(path) == p, 'Baseline study settings/source/environment changed; preserve the old study')
    else:
        save(path, p)
    results, receipts = [], {}
    matrix = [(d, m, h) for d in DTYPES for m in ROWS for h in HIDDEN]
    for number, (dtype, m, h) in enumerate(matrix, 1):
        case = BASE / 'cases' / f'{dtype}_m{m}_h{h}'
        complete = sorted(p for p in case.glob('*/result.json') if p.with_suffix('.sha256').is_file())
        require(len(complete) <= 1, 'Multiple completed results for ' + str(case))
        print(f'\nCase {number}/24: {dtype} [{m},{h}]', flush=True)
        if complete:
            result_path = complete[0]
            result = verify_result(result_path, p, dtype, m, h)
            print('  Verified existing completed case; no timing rerun.', flush=True)
        else:
            folder = case / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            folder.mkdir(parents=True, exist_ok=False)
            result_path = folder / 'result.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--dtype', dtype,
                       '--rows', str(m), '--hidden-size', str(h), '--output', str(result_path)]
            save(folder / 'launch.json', {'command': command, 'protocol_sha256': sha(path)})
            subprocess.run(command, cwd=ROOT, check=True)
            result = verify_result(result_path, p, dtype, m, h)
        results.append(result)
        receipts[str(result_path.relative_to(ROOT))] = sha(result_path)
        save(BASE / 'progress.json', {'completed_cases': number, 'total_cases': 24,
                                      'case_result_sha256': receipts, 'protocol_sha256': sha(path)})
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Source changed while collecting matrix')
    require(gpu_identity() == p['gpu_identity'], 'GPU changed during matrix')
    primary = next(r for r in results if r['dtype'] == 'bfloat16' and r['M'] == 32 and r['H'] == 4096)
    summary = {'protocol': p, 'protocol_sha256': sha(path), 'cases': results,
               'case_result_sha256': receipts, 'primary_case': primary,
               'validated_cases': 24, 'timed_method_batches': 24 * 3 * 6,
               'note': 'Initial kernel, no tuning. Warm CUDA Graph replay; not ordinary operator or full-model latency.'}
    save(BASE / 'baseline_summary.json', summary)
    print('\nInitial kernel — median GPU replay microseconds per invocation')
    print('Dtype       M     H      Eager   Compiled     Triton   Speedup vs compiled')
    lines = ['# Initial kernel: full CUDA Graph matrix', '',
             'Medians of six within-process batch means per method. Speedup = compiled median / Triton median.', '',
             '| Dtype | M | H | Eager us | Compiled us | Triton us | Triton speedup vs compiled |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        s = r['summaries']; ratio = r['speedups_ratio_of_medians']['compiled_over_triton']
        values = [s[k]['median_us'] for k in METHODS]
        print(f'{r["dtype"]:9} {r["M"]:5} {r["H"]:5} {values[0]:10.4f} {values[1]:10.4f} {values[2]:10.4f} {ratio:18.4f}x')
        lines.append(f'| {r["dtype"]} | {r["M"]} | {r["H"]} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {ratio:.4f}x |')
    lines.extend(['', 'The primary case is BF16 [32,4096], retained from the pilot. No best-case result is substituted.', '',
                  'This records GPU execution under warmed replay with fixed input addresses and 64 distinct retained output pairs per method. Compilation, capture, live allocation/dispatch overhead, correctness checking, and data transfers are excluded. Buffers are reused between replays; this is not a measurement of ordinary fresh-allocation Python calls.', '',
                  'All methods use the same inputs, two-output operation, rounding boundary, graph length, replay count, and rotated method order. Correctness is checked before and after timing. Each shape/dtype has one isolated worker and six descriptive timing batches; these are not six independent machine trials or confidence intervals.', '',
                  'Shapes change cache footprint: small inputs/output sets may fit in cache, while large retained output sets do not. No hardware-counter bandwidth claim is made. Keep the full matrix and its raw batch measurements when discussing the result.'])
    (BASE / 'baseline_report.md').write_text('\n'.join(lines) + '\n')
    print('Primary Triton speedup vs compiled:', f'{primary["speedups_ratio_of_medians"]["compiled_over_triton"]:.4f}x')
    print('Saved:', BASE / 'baseline_summary.json')
    print('Saved:', BASE / 'baseline_report.md')
    print('PASS: 24 isolated cases, 432 timing batches, common source/GPU/environment, and pre/post graph correctness verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dtype', choices=DTYPES, help=argparse.SUPPRESS)
    parser.add_argument('--rows', type=int, choices=ROWS, help=argparse.SUPPRESS)
    parser.add_argument('--hidden-size', type=int, choices=HIDDEN, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        require(all(v is not None for v in (args.dtype, args.rows, args.hidden_size, args.output)), 'Worker arguments missing')
        worker(args)
    else:
        controller()


if __name__ == '__main__':
    main()
