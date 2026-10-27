#!/usr/bin/env python3
"""Bounded warp-count search with numerical checks; confirmation is separate."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/day11/warp_tuning_v1'
CANDIDATES = (1, 2, 4, 8, 16)
MIN_IMPROVEMENT = .03


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def select_warps(medians):
    require(set(medians) == set(CANDIDATES) and
            all(math.isfinite(v) and v > 0 for v in medians.values()), 'Invalid candidate medians')
    best = min(CANDIDATES, key=lambda w: (medians[w], w))
    improvement = (medians[4] - medians[best]) / medians[4]
    return best if improvement >= MIN_IMPROVEMENT else 4


def protocol():
    import benchmark_matrix as bm  # Standard-library-only controller helpers.
    baseline_path = ROOT / 'outputs/day11/baseline_v1/baseline_summary.json'
    baseline_protocol_path = ROOT / 'outputs/day11/baseline_v1/protocol.json'
    baseline = read(baseline_path)
    initial = read(baseline_protocol_path)
    require(baseline['validated_cases'] == 24 and baseline['timed_method_batches'] == 432
            and baseline['protocol_sha256'] == sha(baseline_protocol_path), 'Completed initial matrix required')
    for relative, digest in initial['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Initial source changed: ' + relative)
    for relative, digest in baseline['case_result_sha256'].items():
        path = ROOT / relative
        require(sha(path) == digest, 'Initial case result changed')
        case = read(path)
        bm.verify_result(path, initial, case['dtype'], case['M'], case['H'])
    require(bm.gpu_identity() == initial['gpu_identity'], 'GPU differs from initial matrix')
    environment_hash = hashlib.sha256(bm.freeze_text().encode()).hexdigest()
    require(environment_hash == initial['environment_sha256'], 'Environment differs from initial matrix')
    sources = {**initial['source_sha256'],
               str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve()),
               'rmsnorm_lab/launch_variants.py': sha(ROOT / 'rmsnorm_lab/launch_variants.py')}
    orders = [list(CANDIDATES[i:] + CANDIDATES[:i]) for i in range(len(CANDIDATES))]
    return {
        'study': 'day11_warp_tuning_v1', 'initial_baseline_sha256': sha(baseline_path),
        'initial_protocol_sha256': sha(baseline_protocol_path),
        'dtypes': initial['dtypes'], 'M': initial['M'], 'H': initial['H'], 'seed': 123,
        'eps': initial['eps'], 'warp_candidates': list(CANDIDATES), 'baseline_warps': 4,
        'selection_rule': 'Keep four warps unless the fastest candidate median improves it by at least 3 percent; exact candidate ties choose the smaller warp count',
        'minimum_relative_improvement': MIN_IMPROVEMENT,
        'captured_calls_per_replay': 64, 'replays_per_batch': 100, 'batches_per_candidate': 5,
        'candidate_orders': orders, 'warmup_calls': 10, 'warmup_replays': 5,
        'mixed_warmup_seconds': 1., 'correctness_seeds': list(range(10)),
        'boundary_patterns': ['zero', 'cancel', 'halfway_up', 'halfway_down'],
        'unchanged': 'kernel body, residual rounding, output contract, fixed tolerances, block size, one stage, FP fusion disabled',
        'gpu_identity': initial['gpu_identity'], 'environment_sha256': environment_hash,
        'packages': initial['packages'], 'source_sha256': sources,
        'scope': 'Configuration search estimates. A frozen selected policy needs separate validation and independent performance confirmation.',
    }


def worker(args):
    sys.path.insert(0, str(ROOT))
    import torch
    import benchmark_pilot as bp
    import benchmark_matrix as bm
    from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
    from rmsnorm_lab.launch_variants import launch_variant_unchecked, WARP_CANDIDATES
    from check_reference import oracle
    from check_compiled_reference import check_outputs
    from validate_triton import random_inputs

    bp.require_exclusive_gpu()
    require(torch.cuda.is_available(), 'CUDA unavailable')
    setup_pids = bp.active_gpu_pids()
    require(len(setup_pids) <= 1, 'Multiple GPU processes appeared')
    allowed = setup_pids | {os.getpid()}
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    for name in ('CUDA_LAUNCH_BLOCKING', 'PYTORCH_NO_CUDA_MEMORY_CACHING',
                 'TRITON_INTERPRET', 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE'):
        require(os.environ.get(name, '') in ('', '0'), 'Debug setting active: ' + name)
    p = read(BASE / 'protocol.json')
    require(tuple(p['warp_candidates']) == WARP_CANDIDATES == CANDIDATES, 'Candidate set differs')
    require(bp.CAPTURED_CALLS == 64 and bm.gpu_identity() == p['gpu_identity'], 'Timing helper or GPU differs')
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Source changed: ' + relative)
    freeze = bm.freeze_text()
    require(hashlib.sha256(freeze.encode()).hexdigest() == p['environment_sha256'], 'Environment differs')
    contract = read(ROOT / 'specs/contract_v1.json')
    eps, tolerances = p['eps'], contract['output_tolerances'][args.dtype]
    require(eps == contract['eps'], 'Contract epsilon differs')
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    functions = {w: partial(launch_variant_unchecked, num_warps=w) for w in CANDIDATES}
    numerical = {w: {'random_checks': 0, 'boundary_checks': 0, 'max_fraction_of_allowed_error': 0.}
                 for w in CANDIDATES}
    before = bp.gpu_snapshot()

    def validate_all(inputs, kind):
        validate_inputs(*inputs, eps)
        saved_inputs = [t.clone() for t in inputs]
        expected = oracle(*inputs, eps)
        eager = residual_rmsnorm_reference(*inputs, eps)
        check_outputs(eager, expected, inputs, tolerances, 'eager/oracle')
        for w, fn in functions.items():
            actual = fn(*inputs, eps)
            check_outputs(actual, expected, inputs, tolerances, f'warps{w}/oracle')
            check_outputs(actual, eager, inputs, tolerances, f'warps{w}/eager')
            a, e = actual[0].cpu().float(), expected[0].float()
            ratio = float(((a - e).abs() / (tolerances['atol'] + tolerances['rtol'] * e.abs())).max().item())
            require(ratio <= 1, 'Candidate exceeds frozen tolerance')
            require(all(torch.equal(a, b) for a, b in zip(inputs, saved_inputs)), 'Candidate mutated inputs')
            numerical[w][kind + '_checks'] += 1
            numerical[w]['max_fraction_of_allowed_error'] = max(numerical[w]['max_fraction_of_allowed_error'], ratio)

    samples = {w: [] for w in CANDIDATES}
    graphs, outputs = {}, {}
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        print('  Checking all five candidates on seeds 0..9...', flush=True)
        for seed in range(10):
            validate_all(random_inputs(args.rows, args.hidden_size, dtype, seed), 'random')
            print(f'  PASS numerical seed {seed}: all five warp counts', flush=True)
        for pattern in p['boundary_patterns']:
            inputs = list(random_inputs(args.rows, args.hidden_size, dtype, 1001))
            x, residual, weight = inputs
            if pattern == 'zero':
                x.zero_(); residual.zero_()
            elif pattern == 'cancel':
                inputs[1] = -x
            else:
                half = 2**-11 if dtype == torch.float16 else 2**-8
                x.fill_(1 if pattern == 'halfway_up' else 1 + 2 * half)
                residual.zero_(); weight.fill_(1)
                residual[:, 0] = half if pattern == 'halfway_up' else -half
            validate_all(tuple(inputs), 'boundary')
        inputs = random_inputs(args.rows, args.hidden_size, dtype, p['seed'])
        saved_inputs = [t.clone() for t in inputs]
        expected = oracle(*inputs, eps)
        for w, fn in functions.items():
            print(f'  Preparing graph for {w} warps (outside timing)...', flush=True)
            graphs[w], outputs[w] = bp.capture(fn, inputs, eps)
            graphs[w].replay(); torch.cuda.synchronize()
            pointers = [t.untyped_storage().data_ptr() for pair in outputs[w] for t in pair]
            require(len(set(pointers)) == 128, 'Captured outputs alias')
            for pair in outputs[w]:
                check_outputs(pair, expected, inputs, tolerances, f'warps{w}/capture')
        for w in CANDIDATES:
            for _ in range(5):
                graphs[w].replay()
        torch.cuda.synchronize()
        deadline = time.perf_counter() + 1.
        while time.perf_counter() < deadline:
            for w in CANDIDATES:
                for _ in range(10):
                    graphs[w].replay()
            torch.cuda.synchronize()
        bp.require_exclusive_gpu(allowed)
        for repetition, order in enumerate(p['candidate_orders'], 1):
            display = {}
            for w in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graphs[w].replay()
                end.record(); end.synchronize()
                elapsed_ms = start.elapsed_time(end)
                us = elapsed_ms * 1000 / (64 * 100)
                require(math.isfinite(us) and us > 0, 'Invalid timing')
                samples[w].append({'repetition': repetition, 'batch_elapsed_ms': elapsed_ms,
                                   'mean_us_per_invocation': us})
                display[w] = us
            print('  Search batch ' + str(repetition) + '/5: ' + ', '.join(
                f'w{w}={display[w]:.4f} us' for w in CANDIDATES), flush=True)
        for w in CANDIDATES:
            for pair in outputs[w]:
                check_outputs(pair, expected, inputs, tolerances, f'warps{w}/post-timing')
        require(all(torch.equal(a, b) for a, b in zip(inputs, saved_inputs)), 'Search mutated inputs')
    bp.require_exclusive_gpu(allowed)
    medians = {w: statistics.median([b['mean_us_per_invocation'] for b in samples[w]]) for w in CANDIDATES}
    selected = select_warps(medians)
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Source changed during search')
    after = bp.gpu_snapshot()
    require({k: after[k] for k in ('uuid', 'name', 'driver')} == p['gpu_identity'], 'GPU changed')
    folder = args.output.parent
    (folder / 'environment.txt').write_text(freeze)
    result = {
        'dtype': args.dtype, 'M': args.rows, 'H': args.hidden_size,
        'protocol_sha256': sha(BASE / 'protocol.json'), 'source_sha256': p['source_sha256'],
        'gpu_identity': p['gpu_identity'], 'environment_sha256': p['environment_sha256'],
        'candidate_numerical_checks': numerical, 'batches': samples,
        'candidate_median_us': medians, 'selected_warps': selected,
        'search_speedup_vs_four_warps': medians[4] / medians[selected],
        'gpu_before': before, 'gpu_after': after,
        'environment_file_sha256': sha(folder / 'environment.txt'),
        'note': 'Search estimates only; selected policy is not yet independently validated/benchmarked',
    }
    save(args.output, result)
    args.output.with_suffix('.sha256').write_text(sha(args.output) + '\n')
    print(f'PASS search case: selected {selected} warps; search estimate vs four warps '
          f'{result["search_speedup_vs_four_warps"]:.4f}x', flush=True)


def verify_result(path, p, dtype, m, h):
    require(path.with_suffix('.sha256').read_text().strip() == sha(path), 'Search result hash differs')
    r = read(path)
    require((r['dtype'], r['M'], r['H']) == (dtype, m, h), 'Search case differs')
    require(r['protocol_sha256'] == sha(BASE / 'protocol.json') and r['source_sha256'] == p['source_sha256']
            and r['gpu_identity'] == p['gpu_identity'] and r['environment_sha256'] == p['environment_sha256'],
            'Search protocol differs')
    require(sha(path.parent / 'environment.txt') == r['environment_file_sha256'], 'Search environment file changed')
    medians = {}
    for w in CANDIDATES:
        batches = r['batches'][str(w)]
        require(len(batches) == 5 and {b['repetition'] for b in batches} == set(range(1, 6)), 'Missing search batches')
        for b in batches:
            require(b['mean_us_per_invocation'] > 0 and math.isfinite(b['mean_us_per_invocation'])
                    and math.isclose(b['mean_us_per_invocation'], b['batch_elapsed_ms'] * 1000 / (64 * 100), rel_tol=1e-12),
                    'Search timing units differ')
        medians[w] = statistics.median([b['mean_us_per_invocation'] for b in batches])
        require(math.isclose(medians[w], r['candidate_median_us'][str(w)], rel_tol=1e-12), 'Search median differs')
        checks = r['candidate_numerical_checks'][str(w)]
        require(checks['random_checks'] == 10 and checks['boundary_checks'] == 4
                and 0 <= checks['max_fraction_of_allowed_error'] <= 1, 'Candidate correctness checks incomplete')
    require(select_warps(medians) == r['selected_warps'], 'Selection rule differs')
    require(math.isclose(medians[4] / medians[r['selected_warps']], r['search_speedup_vs_four_warps'], rel_tol=1e-12),
            'Search speedup differs')
    return r


def controller():
    p = protocol()
    BASE.mkdir(parents=True, exist_ok=True)
    protocol_path = BASE / 'protocol.json'
    if protocol_path.exists():
        require(read(protocol_path) == p, 'Tuning protocol/source/environment changed; preserve the prior study')
    else:
        save(protocol_path, p)
    results, receipts, choices = [], {}, {}
    cases = [(d, m, h) for d in p['dtypes'] for m in p['M'] for h in p['H']]
    require(len(cases) == 24, 'Expected 24 tuning cases')
    for number, (dtype, m, h) in enumerate(cases, 1):
        key = f'{dtype}_m{m}_h{h}'
        case = BASE / 'cases' / key
        complete = sorted(path for path in case.glob('*/result.json') if path.with_suffix('.sha256').is_file())
        require(len(complete) <= 1, 'Multiple completed tuning results')
        print(f'\nTuning case {number}/24: {dtype} [{m},{h}]', flush=True)
        if complete:
            path = complete[0]
            result = verify_result(path, p, dtype, m, h)
            print('  Verified existing completed search case; no rerun.', flush=True)
        else:
            folder = case / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            folder.mkdir(parents=True, exist_ok=False)
            path = folder / 'result.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--dtype', dtype,
                       '--rows', str(m), '--hidden-size', str(h), '--output', str(path)]
            save(folder / 'launch.json', {'command': command, 'protocol_sha256': sha(protocol_path)})
            subprocess.run(command, cwd=ROOT, check=True)
            result = verify_result(path, p, dtype, m, h)
        results.append(result); receipts[str(path.relative_to(ROOT))] = sha(path)
        choices[key] = {'dtype': dtype, 'M': m, 'H': h, 'num_warps': result['selected_warps']}
        save(BASE / 'progress.json', {'completed_cases': number, 'total_cases': 24,
                                     'case_result_sha256': receipts, 'protocol_sha256': sha(protocol_path)})
    for relative, digest in p['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Source changed during tuning')
    policy = {
        'study': 'day11_selected_warp_policy_v1', 'tuning_protocol_sha256': sha(protocol_path),
        'kernel_source_sha256': sha(ROOT / 'rmsnorm_lab/triton_kernel.py'),
        'variant_source_sha256': sha(ROOT / 'rmsnorm_lab/launch_variants.py'),
        'contract_sha256': sha(ROOT / 'specs/contract_v1.json'), 'cases': choices,
        'fallback': 'Original launch_config for shapes outside these 24 entries',
        'selection_rule': p['selection_rule'],
        'independent_policy_validation_complete': False,
        'independent_confirmation_benchmark_complete': False,
    }
    save(BASE / 'selected_policy.json', policy)
    save(BASE / 'tuning_summary.json', {'protocol': p, 'protocol_sha256': sha(protocol_path),
         'cases': results, 'case_result_sha256': receipts, 'selected_policy_sha256': sha(BASE / 'selected_policy.json'),
         'candidate_random_checks': 24 * 5 * 10, 'candidate_boundary_checks': 24 * 5 * 4,
         'timing_batches': 24 * 5 * 5, 'note': 'Search estimates, not confirmed performance gains'})
    print('\nWarp search — GPU replay estimates, not confirmation results')
    print('Dtype       M     H   Warps   Four-warp us  Selected us  Search speedup')
    for r in results:
        med = r['candidate_median_us']; selected = r['selected_warps']
        print(f'{r["dtype"]:9} {r["M"]:5} {r["H"]:5} {selected:7} {med["4"]:14.4f} '
              f'{med[str(selected)]:12.4f} {r["search_speedup_vs_four_warps"]:14.4f}x')
    print('Saved:', BASE / 'selected_policy.json')
    print('Saved:', BASE / 'tuning_summary.json')
    print('PASS: 24 search cases, 1200 candidate random checks, 480 boundary checks, and 600 search timing batches verified.')
    print('Next: validate and memory-check the frozen selected policy, then benchmark it independently.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), help=argparse.SUPPRESS)
    parser.add_argument('--rows', type=int, choices=(1, 32, 2048), help=argparse.SUPPRESS)
    parser.add_argument('--hidden-size', type=int, choices=(768, 1024, 2560, 4096), help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        require(all(v is not None for v in (args.dtype, args.rows, args.hidden_size, args.output)), 'Worker arguments missing')
        worker(args)
    else:
        controller()


if __name__ == '__main__':
    main()
