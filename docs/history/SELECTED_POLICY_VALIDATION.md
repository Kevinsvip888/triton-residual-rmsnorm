# Validate the frozen warp policy

Extract this archive into `/root/autodl-tmp/triton-kernel-lab`.
It adds two files and this note; it does not replace the initial kernel,
launch variants, tuning script, selected policy, or historical results.

## Full correctness check

```bash
cd /root/autodl-tmp/triton-kernel-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
set -o pipefail
python -u scripts/validate_selected_policy.py \
    2>&1 | tee logs/day11_selected_policy_validation.log
```

Expected: 240 random checks, 30 edge checks, and 21 invalid-input checks.
All 24 selected entries execute. Outside those entries, the wrapper keeps
the original launch policy, including eight warps for hidden sizes above
4096. Edge tests cover those fallback paths and masked tails.

The same fixed FP16/BF16 tolerances apply. Checks use both the FP64 oracle
and eager PyTorch, require exact residual rounding, preserve inputs, and
verify independent fresh outputs. They do not retune or measure speed.
The validator checks saved search results and ties this run to their hashes,
the actual policy, original sources, GPU, and package environment.

## Memory check — only after the full check passes

```bash
set -o pipefail
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
compute-sanitizer --tool memcheck --padding 32 \
    --target-processes all --error-exitcode 99 \
    python -u scripts/validate_selected_policy.py --smoke \
    2>&1 | tee logs/day11_selected_policy_memcheck.log
selected_memcheck_status=${PIPESTATUS[0]}
printf '%s\n' "$selected_memcheck_status" \
    | tee logs/day11_selected_policy_memcheck_exit_code.txt
```

Expected: 24 random checks using a fresh seed (2027), all 30 edge checks,
21 invalid-input checks, `ERROR SUMMARY: 0 errors`, and recorded exit code
`0`. A successful numerical receipt alone does not establish a successful
Compute Sanitizer run. Keep both its log and exit-code file.

The smoke check requires a successful full-validation receipt for these
exact sources and policy. The allocator setting affects only the sanitizer
command, so later benchmarks retain the normal allocator.

## Outputs and next step

Receipts are stored under
`outputs/day11/selected_validation_v1/<timestamp>/`, with a SHA256 sidecar.
They identify the actual wrapper and policy; the historical search policy
is not edited to change its original pending-status flags.

After both checks pass, freeze a separate confirmation benchmark comparing
eager, compiled, initial Triton, and selected-policy Triton in the same run.
Search timings are estimates, not confirmed performance gains.

These files were syntax-checked and dispatch/error paths were checked with
CPU mocks during preparation. GPU execution occurs on your remote machine.
