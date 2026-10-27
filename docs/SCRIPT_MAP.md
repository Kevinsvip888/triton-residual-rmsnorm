# Original source map

The 16 original Python files are preserved byte-for-byte. Publication helpers live in `tools/`; they are separate from the experiment code.

| File | Purpose / dependency |
|---|---|
| `rmsnorm_lab/__init__.py` | Package marker |
| `rmsnorm_lab/reference.py` | Shared metadata checks and eager two-output computation |
| `rmsnorm_lab/triton_kernel.py` | Original JIT arithmetic and fixed launch policy; public/unchecked wrappers |
| `rmsnorm_lab/launch_variants.py` | Same JIT arithmetic with explicit warp count |
| `rmsnorm_lab/tuned_kernel.py` | Cached frozen-policy loader, source-hash checks, selected dispatch and fallback |
| `scripts/check_reference.py` | 14 reference cases, seven invalid-input checks, CPU FP64 oracle |
| `scripts/check_compiled_reference.py` | Eight compiled smoke cases; requires current reference receipt |
| `scripts/check_triton.py` | 24 initial-kernel smoke cases; requires compiled smoke receipt |
| `scripts/validate_triton.py` | 240 random, 30 edge, 21 invalid-input checks; shared generators |
| `scripts/validate_compiled_matrix.py` | 240 random plus 32 boundary cases in eight isolated workers; requires initial validation and memcheck evidence |
| `scripts/benchmark_pilot.py` | Fixed BF16 [32,4096] three-method pilot; graph capture and GPU helpers |
| `scripts/benchmark_matrix.py` | Initial 24-case matrix; six batches per method; resumable workers |
| `scripts/inspect_kernel_launches.py` | Six compiled/initial BF16 profiler captures; launch inventory, not latency benchmarking |
| `scripts/tune_warps.py` | Five warp candidates per case; frozen selection and search receipts |
| `scripts/validate_selected_policy.py` | Full selected-wrapper validation; `--smoke` covers all 24 entries plus edges for memcheck |
| `scripts/benchmark_confirmation.py` | Four-method confirmation, eight batches per method, fixed selected policy |

| Publication helper | Purpose |
|---|---|
| `tools/verify_public_snapshot.py` | Standard-library file/source hash checks and recomputation of available saved timing summaries |
| `tools/restore_selected_policy.py` | Copies the published policy into the runtime path for a demo; refuses to replace different bytes |

Computation helpers deliberately separate metadata validation from the timed tensor workload. Use the validated custom wrappers for ordinary calls; use unchecked wrappers only when their documented preconditions have already been checked.
