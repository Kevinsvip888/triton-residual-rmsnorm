# Residual-add + RMSNorm contract, version 1

The machine-readable contract is [specs/contract_v1.json](specs/contract_v1.json). Its original bytes are preserved because experiment protocols pin its hash. Fields describing planned work reflect its creation at Day 8; completed results are documented in the README.

## Computation

For each token row, let `D` be the input dtype and `gamma = weight`:

1. `residual_out = round_D(float32(x) + float32(residual))`.
2. `v = float32(residual_out)`.
3. Compute `mean_square = sum(v * v) / H` in FP32.
4. `output = round_D((v * rsqrt(mean_square + eps)) * float32(gamma))`.
5. Return `(output, residual_out)` in independent fresh storage.

Default `eps=1e-6` is inside the square root. Gamma multiplies directly, without adding one. There is no mean subtraction, bias, or dropout.

The residual rounding boundary is part of the operation. For BF16, `1 + 1/256` is halfway between adjacent values and rounds to `1` under ties-to-even rounding. Retaining an FP32 sum through normalization would implement a different operation. The fused kernel retains the cast while avoiding an intermediate global-memory reload.

## Supported inputs and outputs

| Property | Requirement |
|---|---|
| `x`, `residual` | Contiguous CUDA `[M,H]`, same nonempty shape |
| `weight` | Contiguous CUDA `[H]` |
| Dtype | All FP16 or all BF16 |
| Device | Same CUDA device for every input |
| Dimensions | Positive `M`; initial and tuned Triton implementations require `1 <= H <= 8192` |
| Epsilon | Positive finite Python int/float; bool is rejected |
| Outputs | Contiguous `[M,H]`, same dtype/device; fresh storage for each output |
| Mutation | None |
| Autograd | Forward inference only; validated custom wrappers reject gradient requests when grad mode is enabled |

The mathematical contract permits positive hidden dimensions; the current Triton implementation deliberately supports the subset up to 8192. The eager computation function assumes validated inputs; it does not validate metadata internally. Use `validate_inputs` separately for eager/compiled calls. The public custom wrappers validate metadata; names ending in `_unchecked` require callers to satisfy those preconditions.

Run under `torch.inference_mode()` with CUDA autocast disabled, as in the checkers and benchmarks.

## Numerical domain and checks

Correctness is tested on finite moderate-range inputs without overflow in the rounded sum, FP32 square/reduction, or final output. Extreme overflow and subnormal behavior are outside the initial guarantee. This is an empirically validated implementation, not an exhaustive proof over all representable values.

Residual outputs must match the oracle exactly. Normalized outputs must be finite and satisfy the fixed per-element bounds:

| Dtype | rtol | atol |
|---|---:|---:|
| FP16 | 0.002 | 0.0002 |
| BF16 | 0.01 | 0.001 |

The rule is `abs(actual - expected) <= atol + rtol * abs(expected)`. Tolerances were fixed before writing/timing the Triton kernel and were not relaxed during tuning.

The CPU FP64 oracle independently computes the rounded residual sum, normalizes in FP64, and casts back to the input dtype. The eager and compiled implementations use FP32 normalization; reduction and reciprocal-square-root differences can cause small low-precision output differences. Residual rounding remains exact in the tests.

The selected policy changes launch warps only. It calls the same initial JIT arithmetic, preserves block padding, uses one stage, and disables FP fusion. Padding masks loads/stores, and the reduction divides by the true `H`, not the padded block length.
