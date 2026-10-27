# Frozen-policy confirmation benchmark

Fresh timings after warp selection and validation, with four fixed methods measured in the same worker.

| Dtype | M | H | Eager us | Compiled us | Initial us | Tuned us | Tuned speedup vs compiled | Tuned speedup vs initial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float16 | 1 | 768 | 15.0282 | 1.0044 | 1.0947 | 1.0137 | 0.9908x | 1.0799x |
| float16 | 1 | 1024 | 14.6031 | 1.0104 | 1.1171 | 1.0498 | 0.9625x | 1.0642x |
| float16 | 1 | 2560 | 15.0318 | 1.3377 | 1.3008 | 1.3012 | 1.0281x | 0.9998x |
| float16 | 1 | 4096 | 15.0196 | 1.4048 | 1.5097 | 1.4527 | 0.9670x | 1.0392x |
| float16 | 32 | 768 | 16.0521 | 1.0412 | 1.1725 | 1.0991 | 0.9473x | 1.0668x |
| float16 | 32 | 1024 | 16.4667 | 1.0598 | 1.1995 | 1.1277 | 0.9398x | 1.0637x |
| float16 | 32 | 2560 | 18.8032 | 1.4186 | 1.3956 | 1.3946 | 1.0172x | 1.0007x |
| float16 | 32 | 4096 | 20.9994 | 1.5231 | 1.6040 | 1.5568 | 0.9783x | 1.0303x |
| float16 | 2048 | 768 | 42.6490 | 13.4086 | 6.6616 | 6.6595 | 2.0135x | 1.0003x |
| float16 | 2048 | 1024 | 51.2184 | 17.9347 | 8.8665 | 8.8701 | 2.0219x | 0.9996x |
| float16 | 2048 | 2560 | 162.4908 | 45.6614 | 26.6732 | 26.6525 | 1.7132x | 1.0008x |
| float16 | 2048 | 4096 | 343.8952 | 72.8611 | 67.7232 | 67.5254 | 1.0790x | 1.0029x |
| bfloat16 | 1 | 768 | 15.1050 | 1.0057 | 1.0858 | 1.0157 | 0.9902x | 1.0690x |
| bfloat16 | 1 | 1024 | 14.6607 | 1.0085 | 1.1176 | 1.0777 | 0.9358x | 1.0370x |
| bfloat16 | 1 | 2560 | 15.0344 | 1.2868 | 1.3950 | 1.2961 | 0.9928x | 1.0763x |
| bfloat16 | 1 | 4096 | 15.0239 | 1.4214 | 1.5830 | 1.4515 | 0.9793x | 1.0906x |
| bfloat16 | 32 | 768 | 16.2449 | 1.0456 | 1.1753 | 1.1102 | 0.9418x | 1.0587x |
| bfloat16 | 32 | 1024 | 16.6612 | 1.0897 | 1.2089 | 1.1661 | 0.9345x | 1.0367x |
| bfloat16 | 32 | 2560 | 19.0091 | 1.4010 | 1.5082 | 1.3984 | 1.0019x | 1.0785x |
| bfloat16 | 32 | 4096 | 21.1787 | 1.5250 | 1.7160 | 1.5732 | 0.9694x | 1.0908x |
| bfloat16 | 2048 | 768 | 42.8684 | 13.4029 | 6.6608 | 6.6749 | 2.0080x | 0.9979x |
| bfloat16 | 2048 | 1024 | 51.6826 | 17.8998 | 8.8545 | 8.8722 | 2.0175x | 0.9980x |
| bfloat16 | 2048 | 2560 | 162.5420 | 45.6636 | 26.2185 | 26.2868 | 1.7371x | 0.9974x |
| bfloat16 | 2048 | 4096 | 343.6318 | 72.8584 | 67.6257 | 67.5980 | 1.0778x | 1.0004x |

Primary case: BF16 [32,4096], retained from the pilot. Speedup is baseline median / tuned median; values below 1 mean tuned is slower.

Primary tuned speedup vs compiled: 0.9694x; vs initial Triton: 1.0908x.
Tuned median faster than compiled in 11/24 cases. This count is descriptive, not a statistical test.

Each method uses the same seed-123 inputs, FP16/BF16 rounding contract, two outputs, 64 calls per CUDA Graph replay, and 200 replays per timing batch. Eight batches per method rotate positions twice. One fresh worker per shape/dtype; 768 total timing batches. Compilation and capture are outside timing. Every retained output is checked against the FP64 oracle before and after timing, with inputs preserved.

The warp policy was frozen before these timings and is not updated by this script. Comparisons use this confirmation collection, not old search/compiled measurements. The original baseline and search reports remain historical records.

These are warmed static-address CUDA Graph GPU-event timings. Python dispatch, live allocation costs and ordinary operator-call latency are excluded. Each method retains 64 distinct output pairs; buffers are reused between replays. This is not full-model latency or production throughput.

Eight within-process batch means are descriptive repeats, not eight independent machine trials or a confidence interval. Cache footprint varies with shape and four retained graph output sets. No hardware bandwidth/occupancy claim follows from these timing results.

A frozen hash inventory ties the run to policy, numerical validation, selected-policy Compute Sanitizer log and zero exit code, original validation, source files, GPU identity, and package environment.
