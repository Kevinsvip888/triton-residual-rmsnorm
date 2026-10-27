# Day 9: first fused Triton kernel

The initial implementation uses one Triton program per token row. Its row-wise indexing, masked loads/stores, and parallel reduction follow the general pattern in the [official Triton normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html). This project computes RMSNorm with residual addition, not mean-subtracting LayerNorm, and has no backward implementation.

For each row, it loads `x`, `residual`, and gamma; adds the first two in FP32; explicitly casts the sum to the input dtype using round-to-nearest-even; promotes it back to FP32 for mean-square normalization; and stores both outputs. The residual cast is retained even though no intermediate global-memory reload is needed.

Columns are padded to a power of two, with a minimum block of 32 lanes. Invalid columns load zero and are never stored. The mean divides by the actual hidden size `H`, not the padded block size. This is important for sizes such as 768, 257, and 4097.

The initial implementation is limited to `1 <= H <= 8192`, a subset of the operation's positive-dimension specification that covers the planned main benchmark. It uses four warps up to H=4096 and eight above, with one stage and FP fusion disabled. These are initial settings, not measured optimum choices. There is no autotuning or claimed speedup.

## Run

Place the files into the existing project alongside the Day 8 files:

```bash
cd /root/autodl-tmp/triton-kernel-lab
source /root/autodl-tmp/qwen-inference-lab/.venv-serve/bin/activate
mkdir -p logs outputs
set -o pipefail
python -u scripts/check_triton.py 2>&1 | tee logs/day9_triton_smoke.log
```

The checker requires a successful compiled-reference receipt for the unchanged Day 8 files. It tests 24 cases: nine random shapes plus halfway rounding, zero, and hand cases for each dtype. Both ordinary and non-power-of-two row lengths are included, along with a 2,048-row case. Checks compare against the eager reference and separate CPU FP64 oracle, enforce the frozen tolerances, require exact residual values, and check repeated-call storage and input preservation.

This is an initial smoke test. The full 240-case matrix, adversarial values, invalid-input checks for the new wrapper, memory-safety checks, and speed measurements are later stages. No GPU execution was available during local file preparation; run the checker on the experiment machine before treating the implementation as working.

The validated public entry point is `residual_rmsnorm_triton`. `residual_rmsnorm_triton_unchecked` separates allocation/launch from shared metadata validation for a later fair benchmark; it requires already validated inputs and the same inference-mode preconditions. Neither entry point implements autograd.
