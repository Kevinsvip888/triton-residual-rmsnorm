# Engineering decisions and interpretation

## Preserve the operation before optimizing it

Residual addition followed by RMSNorm is easy to fuse incorrectly: keeping the sum in FP32 removes the contract's FP16/BF16 rounding boundary. The custom kernel explicitly casts the sum back to the input dtype, then promotes that rounded value for FP32 normalization. Both output tensors remain fresh and inputs stay unchanged.

One Triton program processes one row. Power-of-two padding supports a reduction while masks handle hidden sizes such as 768, 257 and 4097. The denominator uses the actual hidden dimension. The original arithmetic remains in `rmsnorm_lab/triton_kernel.py`; launch variants reuse it rather than maintain a second arithmetic implementation.

## Compare against a fused baseline

Eager PyTorch is a useful reference but does not establish a strong optimization baseline. The study also measures `torch.compile` with `backend='inductor', fullgraph=True, dynamic=False, mode='default'`. Fullgraph requests one compiler graph without graph breaks; it alone does not guarantee one GPU kernel.

The reported launch inspection observed one GPU kernel per invocation for both compiled and initial Triton on three BF16 cases. The inspection script is included, but its original receipt/traces were absent from the review archive. Resource/occupancy metadata from that inspection is not evidence from hardware performance counters, and it does not prove the cause of a latency difference.

## Select a small policy, then freeze it

For every dtype/shape in the main matrix, the search checks 1, 2, 4, 8 and 16 warps on the same original arithmetic. It evaluates every candidate numerically before timing. A candidate replaces four warps only when its median improves by at least 3%; this guard is a selection rule, not a statistical confidence test.

The resulting policy changes 14 of 24 entries. All eight M=2048 entries retain four warps. Small-row entries often favor one warp for H<=1024 or more warps at larger H. These observations are specific to this implementation, GPU and workload; they are not a universal launch policy.

The full selected wrapper is subsequently checked, including its fallback for shapes outside the policy, and memory-instrumented. Confirmation uses fresh timing batches after selection. It does not edit policy entries in response to confirmation scores.

The policy's original `independent_*_complete: false` fields record its status at selection time. They are retained to preserve its identity; later receipts reference its hash. They are not live status flags.

## Read the final result honestly

For the fixed BF16 [32,4096] case, tuning reduces replay time from 1.7160 to 1.5732 microseconds, but the compiled baseline is still faster at 1.5250 microseconds. A custom-kernel optimization can improve its own implementation and still lose to an existing compiler.

For M=2048, the custom kernel beats compiled PyTorch in every tested dtype/hidden-size combination, by approximately 1.08–2.02x. The study therefore demonstrates shape-dependent benefits, not a universal replacement for the compiled baseline. Tiny differences in the unchanged-policy cases should not be presented as new algorithmic improvements.

## What timing does and does not measure

Each graph retains 64 distinct pairs of outputs per method. The host allocations and graph construction occur before timing; replays reuse those buffers. All four methods retain their graphs during each confirmation worker. The resulting working set changes with shape, so small and large cases can have different cache behavior.

CUDA events measure the replay interval, which may include GPU scheduling gaps; it is not a sum of profiler kernel durations. The result excludes live Python dispatch/allocation overhead. There is no ordinary Python-call benchmark or model integration benchmark in this release.

Eight within-process batches help show repeatability under the current setup. They do not quantify variation across machines, drivers, independent sessions, or GPUs. No statistical significance, universal tolerance guarantee, bandwidth saturation, measured occupancy limit, or end-to-end LLM speedup is inferred.
