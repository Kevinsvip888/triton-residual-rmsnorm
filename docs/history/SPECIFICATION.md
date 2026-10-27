# Residual-add + RMSNorm: version 1

For each token row, the function adds a residual connection and applies RMS normalization across the hidden dimension. It returns `(output, residual_out)` and does not modify any input.

Let `D` be the input dtype, `M` the number of token rows, `H` the number of values per row, and `gamma` the scaling vector:

1. `residual_out = round_D(float32(x) + float32(residual))`.
2. `v = float32(residual_out)`.
3. `mean_square = mean(v * v, axis=H)` using FP32.
4. `output = round_D((v * rsqrt(mean_square + eps)) * float32(gamma))`.

The default `eps` is **1e-6**, inside the square root. Gamma multiplies directly; it is not `1 + gamma`. There is no bias, subtraction of the mean, or dropout. RMSNorm's general operation is described in the [PyTorch documentation](https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html); the residual rounding and returned residual are additional choices fixed for this project.

## Supported interface

- `x` and `residual`: contiguous CUDA tensors of the same nonempty shape `[M,H]`.
- `weight` (gamma): contiguous CUDA tensor of shape `[H]`.
- All three inputs use the same CUDA device and dtype, either FP16 or BF16.
- Both outputs use fresh storage, the input dtype/device, and contiguous shape `[M,H]`.
- Forward inference only. Backward, arbitrary strides, FP32 inputs, and in-place updates are outside version 1.
- Correctness covers finite inputs without overflow in the intermediate steps or final dtype. Extreme overflow and subnormal handling are outside the initial contract.

Rounding the sum before normalization is essential. For BF16, `1 + 1/256` is halfway between adjacent values and rounds to `1` under round-to-nearest-even. If a fused kernel keeps that sum in FP32 instead, it implements a different operation. Fusion must retain this cast even if it never stores the temporary to global memory.

## Numerical checks

The machine-readable specification is `specs/contract_v1.json`. Residual outputs must match exactly in the supported finite domain. Normalized values allow FP16 `rtol=0.002, atol=0.0002`, and BF16 `rtol=0.01, atol=0.001`. These tolerances are fixed before writing/timing Triton; they allow low-precision output rounding and small reduction/reciprocal-square-root differences. Future changes must be documented as a new contract rather than relaxing tests to make a faster kernel pass.

Today's smoke test compares the eager FP32 reference against a separate CPU FP64 calculation that preserves the same rounding boundary. It also includes hand-computed values, zero rows, dtype-specific halfway rounding, input non-mutation, independent output storage, and invalid-metadata checks. This is the reference check; it is not the planned 240-case Triton validation or a performance result.

## Planned comparisons

The main matrix uses `M={1,32,2048}`, `H={768,1024,2560,4096}`, both dtypes, and ten seeds: 240 random cases, plus edge cases. Eager PyTorch, `torch.compile`, and Triton must compute this same two-output function under inference mode with autocast disabled. Validation and oracle work stay outside timing. Compilation/startup and warmed execution will be measured separately.

The reference contains several PyTorch operations. The Triton implementation will try to reduce launches and memory traffic; the compiled PyTorch baseline may already fuse much of that work. A speedup is a question to measure, not an assumption.
