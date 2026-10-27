#!/usr/bin/env python3
"""Check available publication bytes and timing arithmetic; never imports torch."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def read(path):
    return json.loads(path.read_text())


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Path outside snapshot: ' + relative)
    return path


def close(a, b, label):
    require(isinstance(a, (int, float)) and isinstance(b, (int, float))
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), label)


def batch_values(rows, count, replays):
    require(len(rows) == count and {r['repetition'] for r in rows} == set(range(1, count + 1)),
            'Missing/duplicate repetitions')
    values = []
    for row in rows:
        value = row['mean_us_per_invocation']
        require(math.isfinite(value) and value > 0, 'Invalid saved time')
        close(value, row['batch_elapsed_ms'] * 1000 / (64 * replays), 'Timing units differ')
        values.append(value)
    return values


def check_stats(saved, values):
    expected = dict(median_us=statistics.median(values), mean_us=statistics.mean(values),
                    sample_sd_us=statistics.stdev(values), min_us=min(values), max_us=max(values))
    require(set(saved) == set(expected), 'Unexpected summary fields')
    for key, value in expected.items():
        close(saved[key], value, 'Incorrect statistic: ' + key)


def verify(root=ROOT, verbose=True):
    root = root.resolve()
    inventory = read(root / 'provenance/publication_sha256.json')
    require(inventory and 'README.md' in inventory and 'tools/verify_public_snapshot.py' in inventory,
            'Incomplete publication inventory')
    for relative, digest in inventory.items():
        require(sha(inside(root, relative)) == digest, 'Published file changed: ' + relative)
    original = read(root / 'provenance/original_source_sha256.json')
    require(len(original) == 17, 'Expected 16 original Python files plus the contract')
    for relative, digest in original.items():
        path = inside(root, relative)
        require(sha(path) == digest, 'Original source changed: ' + relative)
        if path.suffix == '.py':
            ast.parse(path.read_text(), str(path))
    contract = read(root / 'specs/contract_v1.json')
    require(contract['version'] == 1 and contract['output_tolerances'] == {
        'float16': {'rtol': 0.002, 'atol': 0.0002},
        'bfloat16': {'rtol': 0.01, 'atol': 0.001}}, 'Contract changed')
    expected_cases = {(d, m, h) for d in ('float16', 'bfloat16')
                      for m in (1, 32, 2048) for h in (768, 1024, 2560, 4096)}
    stages = [
        ('initial', 'baseline_summary.json', ('eager', 'compiled', 'triton'), 6, 200),
        ('warp_tuning', 'tuning_summary.json', ('1', '2', '4', '8', '16'), 5, 100),
        ('confirmation', 'confirmation_summary.json', ('eager', 'compiled', 'initial', 'tuned'), 8, 200),
    ]
    environment_hash = sha(root / 'environment/requirements-experiment.txt')
    datasets = {}
    common_gpu = None
    total_batches = 0
    for stage, name, methods, count, replays in stages:
        base = root / 'results' / stage
        p, summary = read(base / 'protocol.json'), read(base / name)
        require(summary['protocol'] == p and summary['protocol_sha256'] == sha(base / 'protocol.json'),
                'Protocol differs: ' + stage)
        require(p['environment_sha256'] == environment_hash, 'Environment freeze differs')
        require(p['captured_calls_per_replay'] == 64 and p['replays_per_batch'] == replays,
                'Graph/replay length differs')
        if common_gpu is None:
            common_gpu = p['gpu_identity']
        require(p['gpu_identity'] == common_gpu, 'Different archived GPU identity')
        for relative, digest in p['source_sha256'].items():
            require(sha(inside(root, relative)) == digest, 'Archived source pin differs: ' + relative)
        cases = summary['cases']
        require(len(cases) == 24 and {(r['dtype'], r['M'], r['H']) for r in cases} == expected_cases,
                'Missing/duplicate cases: ' + stage)
        case_hashes = list(summary['case_result_sha256'].values())
        require(len(case_hashes) == len(set(case_hashes)) == 24, 'Case reference hashes differ')
        for r in cases:
            # The original worker save function used indent=2 plus one newline.
            # Reconstruct bytes in memory; do not manufacture live output receipts.
            reconstructed = sha_bytes((json.dumps(r, indent=2) + '\n').encode())
            require(reconstructed in case_hashes, 'Embedded case payload differs from original case hash')
            case_hashes.remove(reconstructed)
            require(r['protocol_sha256'] == summary['protocol_sha256']
                    and r['source_sha256'] == p['source_sha256']
                    and r['gpu_identity'] == common_gpu and r['environment_sha256'] == environment_hash,
                    'Per-case identity differs')
            for snapshot in ('gpu_before', 'gpu_after'):
                require({k: r[snapshot][k] for k in ('uuid', 'name', 'driver')} == common_gpu,
                        'Per-case GPU snapshot differs')
            require(set(r['batches']) == set(methods), 'Method/candidate coverage differs')
            medians = {}
            for method in methods:
                values = batch_values(r['batches'][method], count, replays)
                total_batches += len(values)
                medians[method] = statistics.median(values)
                if stage != 'warp_tuning':
                    check_stats(r['summaries'][method], values)
                else:
                    close(r['candidate_median_us'][method], medians[method], 'Search median differs')
                    checks = r['candidate_numerical_checks'][method]
                    require(checks['random_checks'] == 10 and checks['boundary_checks'] == 4
                            and 0 <= checks['max_fraction_of_allowed_error'] <= 1,
                            'Saved candidate check counts differ')
            if stage == 'initial':
                for method in ('eager', 'compiled'):
                    close(r['speedups_ratio_of_medians'][method + '_over_triton'],
                          medians[method] / medians['triton'], 'Initial speedup differs')
            elif stage == 'warp_tuning':
                best = min(methods, key=lambda w: (medians[w], int(w)))
                chosen = int(best) if (medians['4'] - medians[best]) / medians['4'] >= .03 else 4
                require(r['selected_warps'] == chosen, 'Search selection differs')
                close(r['search_speedup_vs_four_warps'], medians['4'] / medians[str(chosen)],
                      'Search speedup differs')
            else:
                for first, second in (('eager', 'initial'), ('compiled', 'initial'), ('eager', 'tuned'),
                                      ('compiled', 'tuned'), ('initial', 'tuned')):
                    close(r['speedups_ratio_of_medians'][first + '_over_' + second],
                          medians[first] / medians[second], 'Confirmation speedup differs')
        require(not case_hashes, 'Unmatched case hashes')
        if stage == 'warp_tuning':
            require((summary['candidate_random_checks'], summary['candidate_boundary_checks'], summary['timing_batches'])
                    == (1200, 480, 600), 'Search totals differ')
            orders = p['candidate_orders']
            canonical = list(p['warp_candidates'])
            require(p['minimum_relative_improvement'] == .03 and p['baseline_warps'] == 4,
                    'Search rule differs')
        else:
            require(summary['validated_cases'] == 24 and summary['timed_method_batches'] == 24 * len(methods) * count,
                    'Timing totals differ')
            require(summary['primary_case'] in cases and
                    (summary['primary_case']['dtype'], summary['primary_case']['M'], summary['primary_case']['H'])
                    == ('bfloat16', 32, 4096), 'Primary case changed')
            orders, canonical = p['method_orders'], list(methods)
        require(len(orders) == count and all(sorted(order) == sorted(canonical) for order in orders),
                'Invalid method order')
        for method in canonical:
            for position in range(len(canonical)):
                require(sum(order[position] == method for order in orders) == count // len(canonical),
                        'Method order is not position-balanced')
        datasets[stage] = (p, summary)
    policy_path = root / 'results/warp_tuning/selected_policy.json'
    policy, policy_hash = read(policy_path), sha(policy_path)
    psearch, search = datasets['warp_tuning']
    pconfirm, confirmation = datasets['confirmation']
    require(search['selected_policy_sha256'] == pconfirm['selected_policy_sha256'] == policy_hash,
            'Selected policy hash differs')
    require(policy['tuning_protocol_sha256'] == sha(root / 'results/warp_tuning/protocol.json'),
            'Policy protocol differs')
    for field, path in (('kernel_source_sha256', 'rmsnorm_lab/triton_kernel.py'),
                        ('variant_source_sha256', 'rmsnorm_lab/launch_variants.py'),
                        ('contract_sha256', 'specs/contract_v1.json')):
        require(policy[field] == sha(root / path), 'Policy source differs')
    require(len(policy['cases']) == 24, 'Policy coverage differs')
    selections = {f"{r['dtype']}_m{r['M']}_h{r['H']}": r['selected_warps'] for r in search['cases']}
    for r in confirmation['cases']:
        key = f"{r['dtype']}_m{r['M']}_h{r['H']}"
        entry = policy['cases'][key]
        require(entry == {'dtype': r['dtype'], 'M': r['M'], 'H': r['H'], 'num_warps': selections[key]},
                'Policy entry differs')
        require(r['selected_policy_sha256'] == policy_hash and r['tuned_launch_config']['matched_policy']
                and r['tuned_launch_config']['num_warps'] == entry['num_warps'], 'Confirmed dispatch differs')
    require(total_batches == 1800, 'Expected 1800 available timing batches')
    if verbose:
        print(f'PASS: {len(inventory)} public file hashes; 16 original Python files and unchanged contract.')
        print('PASS: three archived protocols, matching source/environment/GPU pins, and 72 embedded case-result hashes.')
        print('PASS: 1800 saved timing batches, medians, speedups, fixed primary and frozen policy recomputed.')
        print('Scope: CPU snapshot verification only. Omitted GPU validation/log/trace files were not audited; no CUDA executed.')
    return {'public_files': len(inventory), 'saved_timing_batches': total_batches,
            'original_python_files': 16, 'embedded_case_payloads': 72}


if __name__ == '__main__':
    verify()
