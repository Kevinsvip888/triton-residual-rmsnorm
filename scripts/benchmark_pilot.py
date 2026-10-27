#!/usr/bin/env python3
"""Day 11 fixed-workload CUDA Graph timing pilot; not Python-call latency."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache/triton'))
sys.path.insert(0, str(ROOT))

import torch
from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
from rmsnorm_lab.triton_kernel import residual_rmsnorm_triton_unchecked, launch_config
from check_reference import oracle
from check_compiled_reference import check_outputs
from validate_triton import random_inputs

METHODS = ('eager', 'compiled', 'triton')
COMPILE_CONFIG = {'backend': 'inductor', 'fullgraph': True, 'dynamic': False, 'mode': 'default'}
M, H, SEED = 32, 4096, 123
CAPTURED_CALLS, REPLAYS_PER_BATCH, REPETITIONS = 64, 200, 6


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


def checked_receipts(pattern, counts):
    found = []
    for path in ROOT.glob(pattern):
        data = read(path)
        if (all(len(data.get(key, [])) == count for key, count in counts.items())
                and data.get('file_sha256')
                and all((ROOT / p).is_file() and sha(ROOT / p) == digest
                        for p, digest in data['file_sha256'].items())):
            found.append(path)
    require(found, 'Missing passing validation receipt for the current files: ' + pattern)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(found)}


def gpu_snapshot():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,driver_version,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip()
    require(len(text.splitlines()) == 1, 'Expected a single physical GPU')
    fields = [v.strip() for v in text.split(',')]
    require(len(fields) == 5, 'Unexpected nvidia-smi output')
    return dict(zip(('uuid', 'name', 'driver', 'temperature_C', 'power_W'), fields))


def active_gpu_pids():
    text = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in text.splitlines() if line.strip().isdigit()}


def require_exclusive_gpu(allowed=None):
    require(not (active_gpu_pids() - (allowed if allowed is not None else {os.getpid()})),
            'Other GPU compute processes are active; stop them before benchmarking')


def capture(fn, inputs, eps):
    # Compilation and allocator warm-up happen before capture and outside timing.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            warm_output = fn(*inputs, eps)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    del warm_output
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with torch.cuda.graph(graph):
        for _ in range(CAPTURED_CALLS):
            outputs.append(fn(*inputs, eps))
    # Retain every captured output pair. Each call must get distinct storage.
    return graph, outputs


def summarize_samples(samples):
    summaries = {}
    require(set(samples) == set(METHODS), 'Method coverage differs')
    for method, batches in samples.items():
        values = [x['mean_us_per_invocation'] for x in batches]
        require(len(values) == REPETITIONS and
                {x['repetition'] for x in batches} == set(range(1, REPETITIONS + 1)),
                'Missing or duplicate timing repetitions')
        require(all(math.isfinite(v) and v > 0 for v in values), 'Nonpositive/nonfinite timing')
        summaries[method] = {'median_us': statistics.median(values), 'mean_us': statistics.mean(values),
                             'sample_sd_us': statistics.stdev(values), 'min_us': min(values), 'max_us': max(values)}
    speedups = {baseline + '_over_triton': summaries[baseline]['median_us'] / summaries['triton']['median_us']
                for baseline in ('eager', 'compiled')}
    return summaries, speedups


def main():
    require_exclusive_gpu()
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    # NVML can report host PIDs rather than container PIDs. Start from an idle
    # GPU, then retain the sole compute context observed after our CUDA init.
    setup_pids = active_gpu_pids()
    require(len(setup_pids) <= 1, 'More than one compute process appeared during setup')
    allowed_pids = setup_pids | {os.getpid()}
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    for name in ('CUDA_LAUNCH_BLOCKING', 'PYTORCH_NO_CUDA_MEMORY_CACHING',
                 'TRITON_INTERPRET', 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE'):
        require(os.environ.get(name, '') in ('', '0'), 'Disable debugging setting for timing: ' + name)
    before = gpu_snapshot()
    require_exclusive_gpu(allowed_pids)
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = read(contract_path)
    require(contract['version'] == 1 and contract['eps'] == 1e-6, 'Unexpected contract')
    evidence = {}
    evidence.update(checked_receipts('outputs/day10/*/triton_validation.json',
                    {'main_cases': 240, 'edge_cases': 30, 'invalid_inputs_rejected': 21}))
    compiled_receipts = checked_receipts('outputs/day10/*/compiled_matrix_validation.json',
                                         {'main_cases': 240, 'boundary_cases': 32})
    evidence.update(compiled_receipts)
    for relative in compiled_receipts:
        data = read(ROOT / relative)
        require(data['compile_config'] == COMPILE_CONFIG, 'Validated compile settings differ')
        require(data['contract_sha256'] == sha(contract_path), 'Validated contract differs')
        for p, digest in data['saved_memcheck_evidence_sha256'].items():
            require(sha(ROOT / p) == digest, 'Saved memcheck evidence changed')
            evidence[p] = digest
    paths = [Path(__file__).resolve(), ROOT / 'rmsnorm_lab/reference.py',
             ROOT / 'rmsnorm_lab/triton_kernel.py', ROOT / 'scripts/check_reference.py',
             ROOT / 'scripts/check_compiled_reference.py', ROOT / 'scripts/validate_triton.py',
             ROOT / 'scripts/validate_compiled_matrix.py', contract_path]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable],
                                     text=True, stderr=subprocess.PIPE)
    orders = [list(METHODS[i % 3:] + METHODS[:i % 3]) for i in range(REPETITIONS)]
    protocol = {
        'study': 'day11_graph_pilot_v1', 'shape': [M, H], 'dtype': 'bfloat16', 'seed': SEED,
        'eps': contract['eps'], 'compile_config': COMPILE_CONFIG, 'triton_config': launch_config(H),
        'captured_calls_per_replay': CAPTURED_CALLS, 'replays_per_batch': REPLAYS_PER_BATCH,
        'repetitions_per_method': REPETITIONS, 'method_orders': orders,
        'warmup_calls_per_method': 10, 'warmup_replays_per_method': 5,
        'mixed_graph_warmup_seconds': 1.,
        'metric': 'CUDA event elapsed microseconds / (captured calls * graph replays)',
        'aggregation': 'median of six batch means per method; speedup is ratio of medians',
        'outputs': '64 retained distinct output pairs per method; storage reused across graph replays',
        'cache_scope': 'repeated static inputs and output buffers, no explicit cache flush',
        'excluded': ['compilation', 'capture', 'Python function dispatch', 'live allocator overhead',
                     'validation', 'CPU/GPU copies', 'oracle computation'],
        'gpu_identity': {k: before[k] for k in ('uuid', 'name', 'driver')},
        'packages': {p: version(p) for p in ('torch', 'triton')}, 'cuda_build': torch.version.cuda,
        'environment_sha256': hashlib.sha256(freeze.encode()).hexdigest(), 'source_sha256': sources,
        'scope': 'one-workload GPU replay pilot, not end-to-end application performance',
    }
    parent = ROOT / 'outputs/day11'
    parent.mkdir(parents=True, exist_ok=True)
    frozen = parent / 'pilot_protocol_v1.json'
    if frozen.exists():
        require(read(frozen) == protocol, 'Pilot protocol/environment/source changed; preserve the prior study')
    else:
        save(frozen, protocol)
    folder = parent / ('pilot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(exist_ok=False)
    (folder / 'environment.txt').write_text(freeze)
    save(folder / 'protocol.json', protocol)
    save(folder / 'evidence_hashes.json', evidence)
    save(folder / 'gpu_before.json', before)
    samples = {method: [] for method in METHODS}
    graphs, outputs = {}, {}
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        inputs = random_inputs(M, H, torch.bfloat16, SEED)
        validate_inputs(*inputs, contract['eps'])
        saved_inputs = [v.clone() for v in inputs]
        expected = oracle(*inputs, contract['eps'])
        methods = {'eager': residual_rmsnorm_reference,
                   'compiled': torch.compile(residual_rmsnorm_reference, **COMPILE_CONFIG),
                   'triton': residual_rmsnorm_triton_unchecked}
        for method in METHODS:
            print('Preparing and capturing ' + method + ' (not timed)...', flush=True)
            graphs[method], outputs[method] = capture(methods[method], inputs, contract['eps'])
            graphs[method].replay()
            torch.cuda.synchronize()
            pointers = [v.untyped_storage().data_ptr() for pair in outputs[method] for v in pair]
            require(len(set(pointers)) == 2 * CAPTURED_CALLS, 'Captured calls alias live output storage')
            for pair in outputs[method]:
                check_outputs(pair, expected, inputs, contract['output_tolerances']['bfloat16'], method + '/graph')
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
        require_exclusive_gpu(allowed_pids)
        for repetition, order in enumerate(orders, 1):
            for method in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(REPLAYS_PER_BATCH):
                    graphs[method].replay()
                end.record()
                end.synchronize()
                elapsed_ms = start.elapsed_time(end)
                us = elapsed_ms * 1000 / (CAPTURED_CALLS * REPLAYS_PER_BATCH)
                require(math.isfinite(us) and us > 0, 'Invalid timing measurement')
                samples[method].append({'repetition': repetition, 'batch_elapsed_ms': elapsed_ms,
                                        'mean_us_per_invocation': us})
                print(f'Rep {repetition}/{REPETITIONS}: {method:8} {us:.4f} us/call', flush=True)
        for method in METHODS:
            for pair in outputs[method]:
                check_outputs(pair, expected, inputs, contract['output_tolerances']['bfloat16'], method + '/post-timing')
        require(all(torch.equal(a, b) for a, b in zip(inputs, saved_inputs)), 'Timing mutated inputs')
    require_exclusive_gpu(allowed_pids)
    after = gpu_snapshot()
    require({k: after[k] for k in ('uuid', 'name', 'driver')} == protocol['gpu_identity'], 'GPU identity changed')
    for relative, digest in {**sources, **evidence}.items():
        require(sha(ROOT / relative) == digest, 'Source/evidence changed during timing: ' + relative)
    summaries, speedups = summarize_samples(samples)
    save(folder / 'gpu_after.json', after)
    report = {'protocol_sha256': sha(frozen), 'protocol': protocol, 'summaries': summaries,
              'batches': samples, 'speedups_ratio_of_medians': speedups,
              'validation_and_memcheck_receipts': evidence,
              'gpu_compute_pids_at_setup': sorted(setup_pids),
              'gpu_before': before, 'gpu_after': after,
              'note': 'GPU replay pilot only. Six batches are descriptive repetitions, not independent trials per invocation.'}
    save(folder / 'pilot_results.json', report)
    print('\nBF16 [32,4096] — warm CUDA Graph replay')
    print('Method       Median us      Min us      Max us')
    for method in METHODS:
        s = summaries[method]
        print(f'{method:10} {s["median_us"]:10.4f} {s["min_us"]:11.4f} {s["max_us"]:11.4f}')
    print(f'Triton speedup vs eager:    {speedups["eager_over_triton"]:.4f}x')
    print(f'Triton speedup vs compiled: {speedups["compiled_over_triton"]:.4f}x')
    print('Saved:', folder / 'pilot_results.json')
    print('PASS: all three captured methods checked before/after timing; common workload and six repetitions verified.')
    print('Scope: warmed GPU replay time; not ordinary Python-call latency or a full-matrix result.')


if __name__ == '__main__':
    main()
