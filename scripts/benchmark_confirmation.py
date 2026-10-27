#!/usr/bin/env python3
"""Independent post-selection confirmation. No candidate search or policy changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/day11/confirmation_v1'
METHODS = ('eager', 'compiled', 'initial', 'tuned')
DTYPES, ROWS, HIDDEN = ('float16', 'bfloat16'), (1, 32, 2048), (768, 1024, 2560, 4096)
REPETITIONS = 8
CAPTURED_CALLS, REPLAYS_PER_BATCH = 64, 200

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

def selected_evidence():
    """Bind the new benchmark to the exact successful numerical/memcheck run."""
    policy_path = ROOT / 'outputs/day11/warp_tuning_v1/selected_policy.json'
    policy = read(policy_path)
    policy_hash = sha(policy_path)
    log = ROOT / 'logs/day11_selected_policy_memcheck.log'
    exit_file = ROOT / 'logs/day11_selected_policy_memcheck_exit_code.txt'
    text = log.read_text(errors='replace')
    require(exit_file.read_text().strip() == '0', 'Selected-policy memcheck exit code is not zero')
    summaries = re.findall(r'ERROR SUMMARY:\s*(\d+) errors', text)
    require(summaries and all(int(v) == 0 for v in summaries), 'Selected-policy memcheck is missing or has errors')
    require('PASS: 24 selected-policy random cases, 30 edge cases, 21 invalid-input checks.' in text,
            'Selected-policy smoke completion missing from memcheck log')
    saved = re.findall(r'^Saved: (.+/smoke_validation\.json)\s*$', text, flags=re.MULTILINE)
    require(len(saved) == 1, 'Expected one smoke receipt in the memcheck log')
    smoke_path = Path(saved[0]).resolve()
    require(smoke_path.is_relative_to(ROOT / 'outputs/day11/selected_validation_v1'),
            'Smoke receipt outside the expected directory')
    smoke = read(smoke_path)
    full_paths = smoke.get('prior_full_validation_sha256', {})
    require(full_paths, 'Smoke receipt has no preceding full validation')
    sources = smoke['file_sha256']
    require(set(('rmsnorm_lab/tuned_kernel.py', 'rmsnorm_lab/triton_kernel.py',
                 'rmsnorm_lab/launch_variants.py', 'scripts/validate_selected_policy.py',
                 'specs/contract_v1.json')).issubset(sources), 'Selected validation source inventory incomplete')
    evidence = {str(policy_path.relative_to(ROOT)): policy_hash,
                str(log.relative_to(ROOT)): sha(log), str(exit_file.relative_to(ROOT)): sha(exit_file),
                str(smoke_path.relative_to(ROOT)): sha(smoke_path)}
    paths = [(smoke_path, 24, 'smoke')]
    for relative, digest in full_paths.items():
        path = ROOT / relative
        require(sha(path) == digest, 'Full validation changed since memcheck')
        paths.append((path, 240, 'full'))
    for path, count, mode in paths:
        r = read(path)
        require(path.with_suffix('.sha256').read_text().strip() == sha(path), 'Validation receipt hash differs')
        require(r['mode'] == mode and len(r['main_cases']) == count and len(r['edge_cases']) == 30
                and len(r['invalid_inputs_rejected']) == 21, 'Selected validation counts differ')
        require(r['evidence'] == smoke['evidence'] and r['evidence']['policy_sha256'] == policy_hash
                and r['file_sha256'] == sources, 'Selected validation source/policy differs')
        require(set(r['executed_policy_entries']) == set(policy['cases']) and len(policy['cases']) == 24,
                'Validation did not cover all selected entries')
        for key, choice in policy['cases'].items():
            cfg = r['executed_policy_entries'][key]
            require(cfg['matched_policy'] and cfg['num_warps'] == choice['num_warps'],
                    'Validated dispatch differs from the policy')
        evidence[str(path.relative_to(ROOT))] = sha(path)
        evidence[str(path.with_suffix('.sha256').relative_to(ROOT))] = sha(path.with_suffix('.sha256'))
    require(smoke['allocator_debug_setting'] == '1', 'Expected memcheck allocator setting')
    for relative, digest in sources.items():
        require(sha(ROOT / relative) == digest, 'Validated source changed: ' + relative)
    for relative in ('protocol.json', 'tuning_summary.json'):
        path = policy_path.parent / relative
        evidence[str(path.relative_to(ROOT))] = sha(path)
    require(smoke['evidence']['tuning_protocol_sha256'] == sha(policy_path.parent / 'protocol.json')
            and smoke['evidence']['tuning_summary_sha256'] == sha(policy_path.parent / 'tuning_summary.json'),
            'Search evidence changed since validation')
    return sources, evidence, policy


def protocol():
    import tune_warps as tuning
    saved_tuning = ROOT / 'outputs/day11/warp_tuning_v1/protocol.json'
    search_protocol = tuning.protocol()  # Re-checks original results, source, GPU and environment; no timing.
    require(read(saved_tuning) == search_protocol, 'Search protocol/source changed')
    sources, evidence, policy = selected_evidence()
    require(all(sources.get(k) == v for k, v in search_protocol['source_sha256'].items()),
            'Selected validation does not bind the search sources')
    require(policy['tuning_protocol_sha256'] == sha(saved_tuning), 'Policy/search mismatch')
    initial = read(ROOT / 'outputs/day11/baseline_v1/protocol.json')
    sources = {**sources, str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())}
    orders = [list(METHODS[i % 4:] + METHODS[:i % 4]) for i in range(REPETITIONS)]
    return {
        'study': 'day11_frozen_policy_confirmation_v1', 'methods': list(METHODS),
        'dtypes': list(DTYPES), 'M': list(ROWS), 'H': list(HIDDEN), 'seed': 123,
        'primary_case': {'dtype': 'bfloat16', 'M': 32, 'H': 4096},
        'eps': initial['eps'], 'compile_config': initial['compile_config'],
        'captured_calls_per_replay': CAPTURED_CALLS, 'replays_per_batch': REPLAYS_PER_BATCH,
        'repetitions_per_method': REPETITIONS, 'method_orders': orders,
        'warmup_calls_per_method': 10, 'warmup_replays_per_method': 5, 'mixed_graph_warmup_seconds': 1.,
        'gpu_identity': initial['gpu_identity'], 'environment_sha256': initial['environment_sha256'],
        'packages': initial['packages'], 'cuda_build': initial['cuda_build'],
        'source_sha256': sources, 'selected_policy_sha256': sha(ROOT / 'outputs/day11/warp_tuning_v1/selected_policy.json'),
        'frozen_evidence_sha256': evidence,
        'outputs': initial['outputs'], 'cache_scope': initial['cache_scope'],
        'metric': 'Warm CUDA Graph GPU-event time per logical invocation, microseconds',
        'aggregation': 'Median of eight within-process batch means; speedups are ratios of method medians',
        'excluded': initial['excluded'],
        'scope': 'Fresh post-selection timings of four fixed methods. No retuning; not ordinary Python-call or full-model latency.',
        'worker_policy': 'One fresh sequential process per shape/dtype; cyclic order gives every method every position twice',
    }


def summarize_samples(samples):
    require(set(samples) == set(METHODS), 'Method coverage differs')
    summaries = {}
    for method in METHODS:
        batches = samples[method]
        require(len(batches) == REPETITIONS and
                {v['repetition'] for v in batches} == set(range(1, REPETITIONS + 1)), 'Missing timing batches')
        values = [v['mean_us_per_invocation'] for v in batches]
        require(all(math.isfinite(v) and v > 0 for v in values), 'Invalid timing')
        summaries[method] = dict(median_us=statistics.median(values), mean_us=statistics.mean(values),
                                 sample_sd_us=statistics.stdev(values), min_us=min(values), max_us=max(values))
    ratios = {a + '_over_' + b: summaries[a]['median_us'] / summaries[b]['median_us']
              for a, b in (('eager','initial'), ('compiled','initial'), ('eager','tuned'),
                           ('compiled','tuned'), ('initial','tuned'))}
    return summaries, ratios


def worker(args):
    sys.path.insert(0, str(ROOT))
    import torch
    import benchmark_pilot as bp
    from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
    from rmsnorm_lab.triton_kernel import residual_rmsnorm_triton_unchecked, launch_config
    from rmsnorm_lab.tuned_kernel import (residual_rmsnorm_tuned_unchecked, selected_launch_config,
                                         policy_sha256)
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
            and bp.REPLAYS_PER_BATCH == 200, 'Pilot capture helper settings changed')
    require(p['methods'] == list(METHODS) and p['repetitions_per_method'] == REPETITIONS
            and p['captured_calls_per_replay'] == CAPTURED_CALLS
            and p['replays_per_batch'] == REPLAYS_PER_BATCH, 'Confirmation settings differ')
    require(policy_sha256() == p['selected_policy_sha256'], 'Loaded policy differs')
    require(gpu_identity() == p['gpu_identity'], 'Worker GPU differs')
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Worker source differs: ' + relative)
    freeze = freeze_text()
    require(hashlib.sha256(freeze.encode()).hexdigest() == p['environment_sha256'], 'Worker environment differs')
    contract = read(ROOT / 'specs/contract_v1.json')
    require(contract['eps'] == p['eps'], 'Contract epsilon changed')
    evidence = dict(p['frozen_evidence_sha256'])
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
                   'initial': residual_rmsnorm_triton_unchecked,
                   'tuned': residual_rmsnorm_tuned_unchecked}
        for method in METHODS:
            print('  Preparing/capturing ' + method + ' (outside timing)...', flush=True)
            graphs[method], outputs[method] = bp.capture(methods[method], inputs, p['eps'])
            graphs[method].replay()
            torch.cuda.synchronize()
            pointers = [v.untyped_storage().data_ptr() for pair in outputs[method] for v in pair]
            require(len(set(pointers)) == 128, 'Captured invocations alias live outputs')
            for pair in outputs[method]:
                check_outputs(pair, expected, inputs, contract['output_tolerances'][args.dtype], method + '/capture')
        all_pointers = [v.untyped_storage().data_ptr() for pairs in outputs.values() for pair in pairs for v in pair]
        require(len(set(all_pointers)) == len(METHODS) * 128, 'Methods share output storage')
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
            print('  Rep ' + str(repetition) + '/8: ' + ', '.join(
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
    summaries, speedups = summarize_samples(samples)
    save(folder / 'gpu_after.json', after)
    result = {
        'dtype': args.dtype, 'M': args.rows, 'H': args.hidden_size,
        'protocol_sha256': sha(BASE / 'protocol.json'), 'source_sha256': p['source_sha256'],
        'gpu_identity': p['gpu_identity'], 'environment_sha256': p['environment_sha256'],
        'initial_launch_config': launch_config(args.hidden_size),
        'tuned_launch_config': selected_launch_config(dtype, args.rows, args.hidden_size),
        'selected_policy_sha256': policy_sha256(), 'seed': p['seed'],
        'summaries': summaries, 'speedups_ratio_of_medians': speedups, 'batches': samples,
        'validation_and_memcheck_receipts': evidence,
        'gpu_before': before, 'gpu_after': after, 'gpu_compute_pids_at_setup': sorted(setup_pids),
        'run_file_sha256': {str(path.relative_to(ROOT)): sha(path) for path in (
            folder / 'environment.txt', folder / 'gpu_before.json', folder / 'gpu_after.json')},
        'note': 'Eight descriptive timing batches in one process; fixed policy; static-input GPU replay only',
    }
    save(args.output, result)
    args.output.with_suffix('.sha256').write_text(sha(args.output) + '\n')
    print(f'PASS case: {args.dtype} [{args.rows},{args.hidden_size}]; '
          f'tuned/compiled speedup={speedups["compiled_over_tuned"]:.4f}x; '
          f'tuned/initial speedup={speedups["initial_over_tuned"]:.4f}x', flush=True)

def verify_result(path, p, dtype, m, h):
    require(path.with_suffix('.sha256').read_text().strip() == sha(path), 'Case result hash differs')
    r = read(path)
    require((r['dtype'], r['M'], r['H']) == (dtype, m, h), 'Case identity differs')
    require(r['protocol_sha256'] == sha(BASE / 'protocol.json') and r['source_sha256'] == p['source_sha256']
            and r['gpu_identity'] == p['gpu_identity'] and r['environment_sha256'] == p['environment_sha256']
            and r['selected_policy_sha256'] == p['selected_policy_sha256'], 'Case protocol differs')
    for relative, digest in {**r['run_file_sha256'], **r['validation_and_memcheck_receipts']}.items():
        require(sha(ROOT / relative) == digest, 'Case evidence changed: ' + relative)
    for relative, digest in p['frozen_evidence_sha256'].items():
        require(r['validation_and_memcheck_receipts'].get(relative) == digest, 'Missing frozen evidence')
    policy = read(ROOT / 'outputs/day11/warp_tuning_v1/selected_policy.json')
    cfg = r['tuned_launch_config']
    selected = policy['cases'][f'{dtype}_m{m}_h{h}']['num_warps']
    require(cfg['matched_policy'] and cfg['num_warps'] == selected, 'Tuned launch differs from selected policy')
    block = max(32, 1 << (h-1).bit_length())
    expected_initial = dict(block_size=block, num_warps=4, num_stages=1, enable_fp_fusion=False)
    require(r['initial_launch_config'] == expected_initial and cfg == {
        **expected_initial, 'num_warps': selected, 'matched_policy': True}, 'Launch settings differ')
    for method in METHODS:
        for b in r['batches'][method]:
            require(math.isclose(b['mean_us_per_invocation'], b['batch_elapsed_ms'] * 1000 /
                                 (CAPTURED_CALLS * REPLAYS_PER_BATCH), rel_tol=1e-12), 'Timing units differ')
    summaries, ratios = summarize_samples(r['batches'])
    require(summaries == r['summaries'] and ratios == r['speedups_ratio_of_medians'], 'Timing summary differs')
    return r


def controller():
    p = protocol()
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / 'protocol.json'
    if path.exists():
        require(read(path) == p, 'Confirmation settings/source/environment/evidence changed; preserve the old study')
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
               'validated_cases': 24, 'timed_method_batches': 24 * 4 * REPETITIONS,
               'note': 'Independent post-selection timings; fixed policy; warm GPU replay only.'}
    save(BASE / 'confirmation_summary.json', summary)
    print('\nConfirmed timings — median GPU replay microseconds per invocation')
    print('Dtype       M     H      Eager   Compiled    Initial      Tuned   Tuned/compiled   Tuned/initial')
    lines = ['# Frozen-policy confirmation benchmark', '',
             'Fresh timings after warp selection and validation, with four fixed methods measured in the same worker.', '',
             '| Dtype | M | H | Eager us | Compiled us | Initial us | Tuned us | Tuned speedup vs compiled | Tuned speedup vs initial |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        s = r['summaries']; ratios = r['speedups_ratio_of_medians']
        values = [s[k]['median_us'] for k in METHODS]
        a, b = ratios['compiled_over_tuned'], ratios['initial_over_tuned']
        print(f'{r["dtype"]:9} {r["M"]:5} {r["H"]:5} ' + ' '.join(f'{v:10.4f}' for v in values) + f' {a:15.4f}x {b:15.4f}x')
        lines.append(f'| {r["dtype"]} | {r["M"]} | {r["H"]} | ' + ' | '.join(f'{v:.4f}' for v in values) + f' | {a:.4f}x | {b:.4f}x |')
    wins = sum(r['speedups_ratio_of_medians']['compiled_over_tuned'] > 1 for r in results)
    pr = primary['speedups_ratio_of_medians']
    lines.extend(['',
        'Primary case: BF16 [32,4096], retained from the pilot. Speedup is baseline median / tuned median; values below 1 mean tuned is slower.', '',
        f'Primary tuned speedup vs compiled: {pr["compiled_over_tuned"]:.4f}x; vs initial Triton: {pr["initial_over_tuned"]:.4f}x.',
        f'Tuned median faster than compiled in {wins}/24 cases. This count is descriptive, not a statistical test.', '',
        'Each method uses the same seed-123 inputs, FP16/BF16 rounding contract, two outputs, 64 calls per CUDA Graph replay, and 200 replays per timing batch. Eight batches per method rotate positions twice. One fresh worker per shape/dtype; 768 total timing batches. Compilation and capture are outside timing. Every retained output is checked against the FP64 oracle before and after timing, with inputs preserved.', '',
        'The warp policy was frozen before these timings and is not updated by this script. Comparisons use this confirmation collection, not old search/compiled measurements. The original baseline and search reports remain historical records.', '',
        'These are warmed static-address CUDA Graph GPU-event timings. Python dispatch, live allocation costs and ordinary operator-call latency are excluded. Each method retains 64 distinct output pairs; buffers are reused between replays. This is not full-model latency or production throughput.', '',
        'Eight within-process batch means are descriptive repeats, not eight independent machine trials or a confidence interval. Cache footprint varies with shape and four retained graph output sets. No hardware bandwidth/occupancy claim follows from these timing results.', '',
        'A frozen hash inventory ties the run to policy, numerical validation, selected-policy Compute Sanitizer log and zero exit code, original validation, source files, GPU identity, and package environment.'])
    (BASE / 'confirmation_report.md').write_text('\n'.join(lines) + '\n')
    print('Primary tuned speedup vs compiled:', f'{pr["compiled_over_tuned"]:.4f}x')
    print('Primary tuned speedup vs initial:', f'{pr["initial_over_tuned"]:.4f}x')
    print('Saved:', BASE / 'confirmation_summary.json')
    print('Saved:', BASE / 'confirmation_report.md')
    print('PASS: 24 isolated cases, 768 confirmation timing batches, fixed selected policy, common GPU/environment and pre/post graph correctness verified.')


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


if __name__ == "__main__":
    main()
