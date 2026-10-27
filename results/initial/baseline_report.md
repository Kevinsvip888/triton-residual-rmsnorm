# Initial kernel: full CUDA Graph matrix

Medians of six within-process batch means per method. Speedup = compiled median / Triton median.

| Dtype | M | H | Eager us | Compiled us | Triton us | Triton speedup vs compiled |
|---|---:|---:|---:|---:|---:|---:|
| float16 | 1 | 768 | 14.8018 | 0.9881 | 1.0757 | 0.9186x |
| float16 | 1 | 1024 | 14.3867 | 0.9942 | 1.0967 | 0.9065x |
| float16 | 1 | 2560 | 14.8066 | 1.3158 | 1.2833 | 1.0253x |
| float16 | 1 | 4096 | 14.7982 | 1.3868 | 1.4874 | 0.9323x |
| float16 | 32 | 768 | 16.0081 | 1.0371 | 1.1688 | 0.8873x |
| float16 | 32 | 1024 | 16.4215 | 1.0582 | 1.1981 | 0.8833x |
| float16 | 32 | 2560 | 18.7623 | 1.4159 | 1.3922 | 1.0170x |
| float16 | 32 | 4096 | 20.9477 | 1.5191 | 1.5998 | 0.9495x |
| float16 | 2048 | 768 | 42.5955 | 13.4014 | 6.6530 | 2.0143x |
| float16 | 2048 | 1024 | 51.1640 | 17.9273 | 8.8684 | 2.0215x |
| float16 | 2048 | 2560 | 162.3919 | 45.6567 | 26.6650 | 1.7122x |
| float16 | 2048 | 4096 | 343.5812 | 72.8333 | 67.5890 | 1.0776x |
| bfloat16 | 1 | 768 | 15.0599 | 1.0037 | 1.0836 | 0.9262x |
| bfloat16 | 1 | 1024 | 14.6158 | 1.0059 | 1.1096 | 0.9066x |
| bfloat16 | 1 | 2560 | 15.0361 | 1.2881 | 1.3916 | 0.9256x |
| bfloat16 | 1 | 4096 | 15.0238 | 1.4230 | 1.5848 | 0.8979x |
| bfloat16 | 32 | 768 | 16.2428 | 1.0444 | 1.1746 | 0.8892x |
| bfloat16 | 32 | 1024 | 16.6124 | 1.0852 | 1.2063 | 0.8996x |
| bfloat16 | 32 | 2560 | 18.9503 | 1.3988 | 1.5068 | 0.9283x |
| bfloat16 | 32 | 4096 | 21.1101 | 1.5229 | 1.7134 | 0.8888x |
| bfloat16 | 2048 | 768 | 42.7964 | 13.4094 | 6.6725 | 2.0097x |
| bfloat16 | 2048 | 1024 | 51.5424 | 17.9316 | 8.8742 | 2.0206x |
| bfloat16 | 2048 | 2560 | 162.3962 | 45.6890 | 26.3095 | 1.7366x |
| bfloat16 | 2048 | 4096 | 343.8199 | 72.8718 | 67.3548 | 1.0819x |

The primary case is BF16 [32,4096], retained from the pilot. No best-case result is substituted.

This records GPU execution under warmed replay with fixed input addresses and 64 distinct retained output pairs per method. Compilation, capture, live allocation/dispatch overhead, correctness checking, and data transfers are excluded. Buffers are reused between replays; this is not a measurement of ordinary fresh-allocation Python calls.

All methods use the same inputs, two-output operation, rounding boundary, graph length, replay count, and rotated method order. Correctness is checked before and after timing. Each shape/dtype has one isolated worker and six descriptive timing batches; these are not six independent machine trials or confidence intervals.

Shapes change cache footprint: small inputs/output sets may fit in cache, while large retained output sets do not. No hardware-counter bandwidth claim is made. Keep the full matrix and its raw batch measurements when discussing the result.
