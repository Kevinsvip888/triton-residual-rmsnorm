# Run the project again

This guide has three routes: inspect the published snapshot with no GPU, run a small CUDA demo using the published policy, or rebuild the complete experiment in a clean checkout. The last route generates new local evidence; it does not restore the original experiment's omitted receipts.

Run each stage only after its predecessor succeeds. Keep the terminal running for GPU jobs. Do not run a vLLM server or another GPU job alongside timing workers.

## 1. Inspect the public snapshot — CPU only

From the repository root:

```bash
python tools/verify_public_snapshot.py
```

This checks public file hashes, source pins, package-freeze identity, archived timing arithmetic, and the selected policy. It does not execute GPU kernels or audit missing traces/logs. It does not require PyTorch to be installed.

Published results are under `results/`; new experiment state is written under ignored `outputs/` and `logs/`. JSON paths referring to `outputs/...` are preserved original experiment paths, not instructions to copy summaries into live runtime state.

## 2. Environment

On the original AutoDL machine, reuse the already-working environment:

```bash
cd /root/autodl-tmp/triton-kernel-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
mkdir -p logs outputs
set -o pipefail

python --version
nvidia-smi
uv pip check
python - <<'PY'
import torch, triton
print('PyTorch:', torch.__version__)
print('Triton:', triton.__version__)
print('PyTorch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    print('BF16 supported:', torch.cuda.is_bf16_supported())
    free, total = torch.cuda.mem_get_info()
    print('Free/total GiB:', free / 2**30, total / 2**30)
PY
```

These commands identify the interpreter, driver/GPU, package consistency and usable CUDA support. The experiment used Python 3.12.15, torch 2.13.0+cu132 and Triton 3.7.1. No upgrade is required on the original machine.

### A fresh environment on another machine

The original environment was reused, not created from a minimal dependency file. The full package record is `environment/requirements-experiment.txt`. If creating a fresh environment, the following is an installation attempt, not a verified clean-machine reproduction:

```bash
uv venv --python 3.12.15 .venv
source .venv/bin/activate
uv pip install --torch-backend=auto \
    -r environment/requirements-experiment.txt
uv pip check
```

This needs accessible package repositories and compatible wheels, including the recorded CUDA PyTorch build. The freeze contains extra serving packages; it does not archive wheels, system tools or the driver. If the exact build cannot be resolved, select its wheel source with the machine provider's instructions rather than silently changing the versions. Record any changed environment as a new experiment. Do not overwrite the published results to make a new environment look like the original.

For a fresh checkout, use its actual path instead of the AutoDL `cd` above. Experiment scripts derive their root from their location. Their within-study source, environment and GPU pins enforce consistency after a new pilot; they do not require a new machine to have the archived physical GPU UUID.

## 3. Small CUDA demo with the published policy

First restore the policy to the path expected by the unchanged selected wrapper:

```bash
python tools/restore_selected_policy.py
```

This verifies the snapshot and copies `results/warp_tuning/selected_policy.json` to `outputs/day11/warp_tuning_v1/selected_policy.json`. Existing identical bytes are accepted; different bytes cause an error. The helper does not install archived benchmark protocols, fake validation receipts, or mark a new benchmark complete.

Then compare a BF16 primary-shape call with eager PyTorch:

```bash
python - <<'PY'
import torch
from rmsnorm_lab.reference import residual_rmsnorm_reference, validate_inputs
from rmsnorm_lab.tuned_kernel import residual_rmsnorm_tuned, selected_launch_config

with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
    torch.manual_seed(123)
    x = torch.randn(32, 4096, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(4096, device='cuda', dtype=x.dtype)
    validate_inputs(x, residual, weight)
    expected = residual_rmsnorm_reference(x, residual, weight)
    actual = residual_rmsnorm_tuned(x, residual, weight)
    assert torch.equal(actual[1], expected[1])
    torch.testing.assert_close(actual[0].float(), expected[0].float(),
                               rtol=0.01, atol=0.001)
    print('Launch:', selected_launch_config(x.dtype, *x.shape))
    print('PASS: one selected-policy CUDA demo; no timing performed.')
PY
```

The first call can compile. This example is a usage demonstration to run on your GPU, not a replacement for the full validation below. The published policy was chosen on an RTX 4090; its performance is not established on another GPU.

## 4. Full experiment from a clean checkout

Use a separate clean checkout with no existing `outputs/` or `logs/` study state. Do not restore the published policy first for this route: tuning generates a new policy. Do not delete your original experiment to create a clean workspace. Keep the same source files, environment and physical GPU through a study.

Activate the intended environment, create output directories, and retain pipeline failure status:

```bash
mkdir -p logs outputs
set -o pipefail
```

### Day 8 — operation and compiled baseline

```bash
python -u scripts/check_reference.py \
    2>&1 | tee logs/day8_reference.log

python -u scripts/check_compiled_reference.py \
    2>&1 | tee logs/day8_compiled_reference.log
```

The first command checks 14 reference cases and seven invalid-input cases against an independent CPU FP64 oracle. It includes a hand calculation, zero rows and halfway residual rounding. The second checks eight compiled cases with the same output contract. Both write timestamped receipts with code/contract hashes; compilation time is not a performance measurement.

### Day 9 — initial Triton smoke tests

```bash
python -u scripts/check_triton.py \
    2>&1 | tee logs/day9_triton_smoke.log
```

Expected: 24 cases across FP16/BF16, including masked tails, halfway rounding, zeros, hand examples and boundary hidden sizes. The script requires the preceding compiled-reference receipt for current source bytes.

### Day 10 — full correctness and memory checks

```bash
python -u scripts/validate_triton.py \
    2>&1 | tee logs/day10_triton_validation.log
```

Expected: 240 random cases (24 dtype/shape combinations × seeds 0–9), 30 edge cases, and 21 invalid-input checks. It checks exact residuals, frozen error bounds, unchanged inputs and independent repeated outputs.

Check that Compute Sanitizer is installed, then memory-instrument the initial smoke suite:

```bash
compute-sanitizer --version

PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
compute-sanitizer --tool memcheck --padding 32 \
    --target-processes all --error-exitcode 99 \
    python -u scripts/check_triton.py \
    2>&1 | tee logs/day10_memcheck.log
memcheck_status=${PIPESTATUS[0]}
printf '%s\n' "$memcheck_status" \
    | tee logs/day10_memcheck_exit_code.txt
```

The per-command allocator setting makes the memory check easier to instrument; it does not change later benchmark settings. `--padding 32` adds allocation padding for checking, `--target-processes all` covers child processes, and `--error-exitcode 99` makes reported tool errors fail the command. Capture `PIPESTATUS[0]` immediately after the pipeline so it records the sanitizer/Python process, not `tee`.

Proceed only with a completed 24-case smoke check, `ERROR SUMMARY: 0 errors`, and saved exit code `0`.

```bash
python -u scripts/validate_compiled_matrix.py \
    2>&1 | tee logs/day10_compiled_matrix.log
```

Expected: 240 compiled random cases and 32 boundary cases in eight isolated workers. Isolation limits accumulated compiler specializations. This command requires current Triton validation plus saved initial memcheck log/exit code. It does not time performance.

### Day 11 — pilot and complete initial matrix

```bash
python -u scripts/benchmark_pilot.py \
    2>&1 | tee logs/day11_pilot.log

python -u scripts/benchmark_matrix.py \
    2>&1 | tee logs/day11_baseline_matrix.log
```

The pilot fixes BF16 `[32,4096]` and establishes the graph timing protocol. The full matrix measures 24 cases, three methods and six batches per method: 432 timing batches. Every batch uses 200 replays of a 64-call graph, excluding preparation and correctness checks from timing.

The matrix controller uses one fresh sequential worker per case. Rerunning its command verifies and skips completed cases; a partial case restarts in a new attempt directory. A source/environment/GPU change causes an error instead of mixing different studies.

Optional launch inspection, as performed before the original search:

```bash
python -u scripts/inspect_kernel_launches.py \
    2>&1 | tee logs/day11_launch_inspection.log
```

This profiles compiled and initial Triton on three BF16 shapes, with 20 ordinary calls per capture. It saves launch/resource metadata and six raw traces. These captures are not benchmark timings; profiler occupancy fields may be estimates.

### Day 11 — search and freeze warp choices

```bash
python -u scripts/tune_warps.py \
    2>&1 | tee logs/day11_warp_tuning.log
```

For each case, this checks 1/2/4/8/16 warps and collects five timing batches per candidate. Search batches use 100 graph replays, distinct from the 200 used in the main benchmarks. Totals: 1,200 candidate random checks, 480 boundary checks and 600 timing batches.

The fastest candidate replaces four warps only if it improves the concurrently measured four-warp median by at least 3%. The policy is saved at `outputs/day11/warp_tuning_v1/selected_policy.json`. Do not edit its historical pending-status flags. Search estimates are not final confirmed gains.

### Day 11 — validate and memory-check the frozen policy

```bash
python -u scripts/validate_selected_policy.py \
    2>&1 | tee logs/day11_selected_policy_validation.log
```

Expected: 240 random, 30 edge and 21 invalid-input checks. Every selected policy entry executes; fallback shapes are checked too. The script verifies the source/policy/search identities.

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
compute-sanitizer --tool memcheck --padding 32 \
    --target-processes all --error-exitcode 99 \
    python -u scripts/validate_selected_policy.py --smoke \
    2>&1 | tee logs/day11_selected_policy_memcheck.log
selected_memcheck_status=${PIPESTATUS[0]}
printf '%s\n' "$selected_memcheck_status" \
    | tee logs/day11_selected_policy_memcheck_exit_code.txt
```

Expected: 24 random cases using fresh seed 2027, all 30 edges, 21 invalid-input checks, zero sanitizer errors and exit code zero. The smoke mode requires the successful full validation for these sources/policy.

### Day 11 — independent confirmation

```bash
python -u scripts/benchmark_confirmation.py \
    2>&1 | tee logs/day11_confirmation.log
```

This compares eager, compiled, initial and tuned methods across the 24 fixed cases. Eight rounds rotate method order, giving 768 timing batches. The primary case remains BF16 `[32,4096]`. All methods are measured afresh; old search times are not used as final performance measurements. No new policy entries are chosen.

The script requires the exact selected-policy memory-check filenames above. It verifies the logged smoke receipt, saved zero exit code, policy and source hashes. It then checks every retained graph output before and after timing. Completed cases resume when the same command is rerun without changing the study inputs.

New final outputs:

```text
outputs/day11/confirmation_v1/confirmation_summary.json
outputs/day11/confirmation_v1/confirmation_report.md
```

### Day 12 — record and review

```bash
uv pip freeze > logs/project2-requirements.txt
python tools/verify_public_snapshot.py
git status --short
```

The freeze records the current environment. The public verifier still checks the published snapshot, not your newly generated output files. Your new GPU run is established by its own final PASS output, receipts, logs and hashes.

Keep a separate backup of `outputs/`, `logs/`, code and the environment record. Those original runtime files are needed for a full later audit; the small public snapshot intentionally omits much of that evidence. GitHub publication is a separate step after reviewing the staged file list.

## Reading speedups and failures

A reported speedup is baseline median / custom median. Above 1 means the custom method took less GPU replay time; below 1 means it took more. A 1.09x speedup is not a 9% reduction in time: reduction is `1 - 1/speedup`.

A first-call compilation pause or graph preparation is expected and is printed separately from timed rounds. A numerical failure, memory error, source hash mismatch, different physical GPU, changed package freeze or missing prior receipt requires investigation. Preserve the original study and failing output; do not relax tolerances or rewrite receipts to continue.
