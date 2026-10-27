#!/usr/bin/env python3
"""Inspect launch structure for three BF16 cases; not a latency benchmark."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.cache/triton'))
sys.path.insert(0, str(ROOT))

import torch
import benchmark_pilot as bp
from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
from rmsnorm_lab.triton_kernel import residual_rmsnorm_triton_unchecked
from check_reference import oracle
from check_compiled_reference import check_outputs
from validate_triton import random_inputs

CASES = ((32, 4096), (2048, 1024), (2048, 4096))
ITERATIONS = 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(trace, iterations):
    kernels = [event for event in trace.get('traceEvents', [])
               if event.get('ph') == 'X'
               and 'kernel' in {part.strip().lower() for part in event.get('cat', '').split(',')}]
    require(kernels, 'No GPU kernel events found; raw trace was preserved')
    grouped = Counter()
    for event in kernels:
        resources = {key: value for key, value in event.get('args', {}).items()
                     if any(word in key.lower() for word in ('grid', 'block', 'register', 'shared', 'occupancy'))}
        key = (event.get('name', 'unknown'), json.dumps(resources, sort_keys=True))
        grouped[key] += 1
    require(len(kernels) % iterations == 0,
            'GPU launch count is not a multiple of profiled invocations; inspect the preserved trace')
    rows = [{'kernel': name, 'launches': count, 'launches_per_invocation': count / iterations,
             'profiler_resource_metadata': json.loads(resources)}
            for (name, resources), count in sorted(grouped.items())]
    return {'profiled_invocations': iterations, 'gpu_kernel_launches': len(kernels),
            'gpu_kernels_per_invocation': len(kernels) / iterations, 'kernels': rows}


def main():
    bp.require_exclusive_gpu()
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    require(torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(),
            'This PyTorch installation does not expose CUDA profiler activity')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    baseline_path = ROOT / 'outputs/day11/baseline_v1/baseline_summary.json'
    baseline = json.loads(baseline_path.read_text())
    protocol_path = ROOT / 'outputs/day11/baseline_v1/protocol.json'
    require(baseline['protocol_sha256'] == sha(protocol_path) and baseline['validated_cases'] == 24,
            'A completed initial matrix is required')
    for relative, digest in baseline['protocol']['source_sha256'].items():
        require(sha(ROOT / relative) == digest, 'Initial source changed: ' + relative)
    require(baseline['protocol']['compile_config'] == bp.COMPILE_CONFIG, 'Compile settings changed')
    gpu = bp.gpu_snapshot()
    require({key: gpu[key] for key in ('uuid', 'name', 'driver')} == baseline['protocol']['gpu_identity'],
            'GPU identity differs from the baseline')
    folder = ROOT / 'outputs/day11' / ('inspection_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    contract_path = ROOT / 'specs/contract_v1.json'
    contract = json.loads(contract_path.read_text())
    compiled = torch.compile(residual_rmsnorm_reference, **bp.COMPILE_CONFIG)
    methods = {'compiled': compiled, 'triton': residual_rmsnorm_triton_unchecked}
    results, trace_hashes = [], {}
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for m, h in CASES:
            inputs = random_inputs(m, h, torch.bfloat16, 123)
            validate_inputs(*inputs, contract['eps'])
            input_snapshots = [value.clone() for value in inputs]
            expected = oracle(*inputs, contract['eps'])
            for method, fn in methods.items():
                name = f'{method}_bf16_m{m}_h{h}'
                print('\nInspecting ' + name + '...', flush=True)
                for _ in range(10):
                    warm = fn(*inputs, contract['eps'])
                torch.cuda.synchronize()
                del warm
                profiled_outputs = []
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False,
                ) as profile:
                    with torch.profiler.record_function(name):
                        for _ in range(ITERATIONS):
                            profiled_outputs.append(fn(*inputs, contract['eps']))
                        torch.cuda.synchronize()
                trace_path = folder / (name + '.trace.json')
                profile.export_chrome_trace(str(trace_path))
                trace_hashes[str(trace_path.relative_to(ROOT))] = sha(trace_path)
                for pair in profiled_outputs:
                    check_outputs(pair, expected, inputs, contract['output_tolerances']['bfloat16'], name)
                require(all(torch.equal(a, b) for a, b in zip(inputs, input_snapshots)), 'Profiling mutated inputs')
                item = inventory(json.loads(trace_path.read_text()), ITERATIONS)
                item.update(method=method, dtype='bfloat16', M=m, H=h,
                            trace_path=str(trace_path.relative_to(ROOT)), trace_sha256=sha(trace_path))
                results.append(item)
                print(f'GPU kernels per invocation: {item["gpu_kernels_per_invocation"]:.0f}')
                for row in item['kernels']:
                    print(f'  {row["launches"]} launches: {row["kernel"]}')
                    print('  Resources:', json.dumps(row['profiler_resource_metadata'], sort_keys=True))
                del profiled_outputs, profile
    sources = {**baseline['protocol']['source_sha256'],
               str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())}
    for relative, digest in sources.items():
        require(sha(ROOT / relative) == digest, 'Source changed during inspection: ' + relative)
    report = {
        'scope': 'ordinary warmed launches for kernel inventory; not the CUDA Graph timing workload',
        'cases': results, 'iterations_per_method_case': ITERATIONS, 'trace_sha256': trace_hashes,
        'source_sha256': sources, 'baseline_summary_sha256': sha(baseline_path),
        'baseline_protocol_sha256': sha(protocol_path), 'compile_config': bp.COMPILE_CONFIG,
        'packages': {p: version(p) for p in ('torch', 'triton')}, 'gpu_snapshot': gpu,
        'resource_note': 'Fields are copied from profiler event metadata when present. Occupancy fields may be estimates, not hardware-counter measurements.',
        'timing_note': 'Do not substitute profiler durations for the unprofiled CUDA Graph measurements.',
        'performance_measured': False,
    }
    path = folder / 'kernel_launch_inventory.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('\nSaved:', path)
    print('PASS: six profiled method/case combinations, GPU launch events, trace hashes, and output correctness verified.')
    print('This is launch-structure evidence, not a new performance benchmark.')


if __name__ == '__main__':
    main()
