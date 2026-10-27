# Triton residual-add + RMSNorm lab

Day 8 establishes the operation contract and eager PyTorch reference. No custom Triton kernel or performance result exists yet.

Read [SPECIFICATION.md](SPECIFICATION.md) for the equations, rounding boundary, supported inputs, and preselected error tolerances. `specs/contract_v1.json` provides the corresponding machine-readable record.

## Environment

Initial environment check on the experiment machine:

| Item | Version / value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Compute capability | 8.9 |
| Python | 3.12.15 |
| PyTorch | 2.13.0+cu132 |
| Triton | 3.7.1 |
| CUDA build reported by PyTorch | 13.2 |

The existing serving environment from Project 1 is reused without upgrading its packages. GPU tests are run on the experiment machine. Publication-side preparation checked syntax and independent scalar rounding examples, not GPU execution.

## Run the reference checks

```bash
cd /root/autodl-tmp/triton-kernel-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
mkdir -p logs outputs
set -o pipefail

python -u scripts/check_reference.py 2>&1 | tee logs/day8_reference.log
```

Expected success: 14 valid reference cases and seven invalid-input checks. The valid cases include random input shapes, non-unit/negative gamma, a hand-computed row, zero rows, and dtype-specific halfway residual rounding. The checker independently normalizes in CPU FP64 after rounding the residual sum to the original dtype. It also checks fresh output storage and input non-mutation.

Each successful run saves a timestamped `outputs/day8/.../reference_checks.json`, including environment information, error measurements, and hashes of the code and contract. Check stdout and the exit code; a partial console log does not establish success.

## Files

| File | Role |
|---|---|
| `rmsnorm_lab/reference.py` | Metadata validator and eager FP32 computation with explicit dtype rounding |
| `scripts/check_reference.py` | GPU reference checks, CPU FP64 oracle, and saved receipt |
| `specs/contract_v1.json` | Versioned operation and tolerance definition |
| `SPECIFICATION.md` | Explanation and planned benchmark scope |

The computation function deliberately does not validate metadata internally. Call `validate_inputs` before dispatch; keep validation, cloning, oracle execution, and correctness checks outside future benchmark timing. Run inference under `torch.inference_mode()` with autocast disabled. The same two-output computation will be used for the eager, compiled, and Triton comparisons.

The full 240-case randomized Triton validation, adversarial/edge tests, compiled-baseline checks, timing methodology, and tuning are later stages. Today's reference smoke tests do not replace them.
