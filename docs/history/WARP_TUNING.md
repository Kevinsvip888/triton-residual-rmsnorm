# Warp-count search

The launch inventory showed one GPU kernel per invocation for both compiled PyTorch and the initial custom kernel. For BF16 H=4096, the compiled kernel used a 512-thread block while the custom kernel used 128 threads. More threads may reduce work/registers per thread, but extra warps also add reduction/coordination costs. Occupancy values copied from profiler metadata are estimates and do not establish a bottleneck.

This search tests **1, 2, 4, 8, and 16 warps**: 32, 64, 128, 256, and 512 threads per block on this NVIDIA GPU. It reuses the original JIT kernel body without editing it. The dtype rounding boundary, block size, one software-pipeline stage, disabled FP fusion, output semantics, and numerical tolerances remain fixed.

## Procedure fixed before search

For each of the 24 shape/dtype cases, one isolated worker:

1. Checks all five variants on the ten main random seeds and four boundary patterns (zero, cancellation, upward halfway, downward halfway).
2. Captures 64 calls per variant, retaining distinct output pairs, using the same graph-capture helper as the initial benchmark.
3. Warms the GPU and collects five batches of 100 replays per candidate. Candidate order rotates so every candidate occupies each position once.
4. Selects the fastest median only if it improves the concurrently measured four-warp median by **at least 3%**. Otherwise it retains four warps. Exact candidate ties choose the smaller warp count when a qualifying improvement exists.

The 3% guard reduces tiny/noisy configuration switches; it is not a statistical confidence threshold. Search and confirmation are separate measurements. Timings in this stage are selection estimates and must not be advertised as independently confirmed speedups. Four warps is the initial default for every main H value (all are at most 4096).

All five candidates must pass correctness; the script stops on a failure rather than relaxing tolerances or silently excluding a candidate. It does not yet run the selected policy through the full additional edge/API suite or memory instrumentation. Those checks and an independent benchmark follow after the policy is frozen.

## Run

```bash
cd /root/autodl-tmp/triton-kernel-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
set -o pipefail
python -u scripts/tune_warps.py 2>&1 | tee logs/day11_warp_tuning.log
```

Output: `outputs/day11/warp_tuning_v1/selected_policy.json` and `tuning_summary.json`, plus per-case raw batches, correctness summaries, source/environment hashes, and GPU snapshots. Rerunning verifies/skips completed cases. An unfinished case restarts in a new attempt directory. Initial benchmark files and original kernel source remain unchanged.

The selected policy is a new experiment artifact. The historical status fields in it record that confirmation was still pending when selection finished; later validation/benchmark receipts should refer to its hash instead of editing those fields in place.
