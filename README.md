# Triton residual-add + RMSNorm lab

A forward-inference GPU kernel study comparing eager PyTorch, `torch.compile`, and a custom Triton kernel on an RTX 4090. The kernel adds a residual connection, preserves its low-precision rounding, and returns both the normalized output and the rounded residual sum.

**Result:** the custom kernel is strongest at 2,048 token rows, reaching **2.02×** the compiled baseline's speed in warmed CUDA Graph replay. Warp tuning improves the preselected BF16 `[32,4096]` case by **1.09×** over the initial kernel, while the tuned kernel still takes about **3.2% longer than compiled PyTorch** for that case. There is no universal custom-kernel win.

## Main result

The primary case was fixed before tuning: **BF16, M=32, H=4096**. Lower time is better.

| Method | Median GPU replay time per invocation |
|---|---:|
| Eager PyTorch | 21.1787 µs |
| Compiled PyTorch | **1.5250 µs** |
| Initial Triton, 4 warps | 1.7160 µs |
| Tuned Triton, 16 warps | 1.5732 µs |

Tuned speedup over initial Triton is **1.0908×**, corresponding to **8.33% less GPU replay time**. Tuned speedup over compiled PyTorch is **0.9694×**: the custom kernel still takes **3.16% longer**. The best large-batch result does not replace this primary result.

At **M=2048**, the tuned kernel beats the compiled baseline in every tested dtype/hidden-size combination:

| Dtype | H | Compiled µs | Tuned µs | Speedup |
|---|---:|---:|---:|---:|
| FP16 | 768 | 13.4086 | 6.6595 | 2.0135× |
| FP16 | 1024 | 17.9347 | 8.8701 | 2.0219× |
| FP16 | 2560 | 45.6614 | 26.6525 | 1.7132× |
| FP16 | 4096 | 72.8611 | 67.5254 | 1.0790× |
| BF16 | 768 | 13.4029 | 6.6749 | 2.0080× |
| BF16 | 1024 | 17.8998 | 8.8722 | 2.0175× |
| BF16 | 2560 | 45.6636 | 26.2868 | 1.7371× |
| BF16 | 4096 | 72.8584 | 67.5980 | 1.0778× |

Compiled PyTorch wins most smaller-row cases. Tuned medians beat it in 11/24 cases overall; some differences are very small, so this count is descriptive. See the [full confirmation table](results/confirmation/confirmation_report.md) and [saved batches](results/confirmation/confirmation_summary.json).

## What was implemented

- A shared operation contract and eager reference, plus a separate CPU FP64 oracle.
- A masked, row-wise Triton kernel with explicit residual rounding and two fresh outputs.
- Correctness and invalid-input checks, followed by Compute Sanitizer memory checks.
- A fixed 24-case benchmark matrix and a bounded warp-count search.
- A frozen shape-specific launch policy, checked before a separate confirmation benchmark.

The arithmetic is unchanged by tuning. Only the number of warps changes for selected dtype/shape entries. Shapes outside the 24-entry policy retain the initial launch settings.

## Operation and interface

For input dtype `D`, with contiguous CUDA inputs `x, residual: [M,H]` and `weight: [H]`:

```python
residual_out = (x.float() + residual.float()).to(D)
v = residual_out.float()
output = ((v * torch.rsqrt((v * v).mean(-1, keepdim=True) + 1e-6))
          * weight.float()).to(D)
return output, residual_out
```

The cast of the sum back to `D` happens **before normalization**, including in the fused kernel. Inputs are FP16 or BF16; `1 <= H <= 8192`; positive `M`; same device and dtype. This is forward inference with fresh outputs and no input mutation. See [SPECIFICATION.md](SPECIFICATION.md) for the supported numerical domain and fixed tolerances.

## Read or run

Verify the public files and recompute the archived timing summaries using only Python's standard library:

```bash
python tools/verify_public_snapshot.py
```

For a CUDA smoke run, environment setup, every stage's commands, and the distinction between archived results and a new experiment, follow [docs/REPRODUCE.md](docs/REPRODUCE.md). The selected wrapper requires a policy file; the guide shows how to restore the published policy for a demo or generate a new one through the full pipeline.

## Experiment setup

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB; compute capability 8.9 |
| Python | 3.12.15, as reported during setup |
| PyTorch | 2.13.0+cu132 |
| Triton | 3.7.1 |
| PyTorch CUDA build | 13.2 |
| NVIDIA driver | 580.76.05 |
| Main matrix | FP16/BF16 × M={1,32,2048} × H={768,1024,2560,4096} |
| Confirmation | 24 isolated workers, four methods, eight timing batches each |

The full reused environment is recorded in [environment/requirements-experiment.txt](environment/requirements-experiment.txt). It includes Project 1 serving dependencies beyond this kernel's direct needs. See [environment/README.md](environment/README.md).

## Measurement scope

The numbers above are **warmed CUDA Graph GPU-event times**, not ordinary Python-call latency, fresh-allocation latency, or full-model throughput. Each graph contains 64 logical calls with distinct retained output pairs. Each timing batch replays it 200 times; outputs reuse their addresses between replays. Compilation, graph capture, oracle checks, and data transfers are outside timing.

All four methods run in the same per-case process with matching inputs and output semantics. Method order rotates so each method occupies each position twice. Eight within-process batch means are descriptive repeats, not eight independent machine trials or a confidence interval. Cache footprint varies by shape and retained output size.

The initial study, search, and confirmation remain separate records. Search tested 1/2/4/8/16 warps and retained four unless a candidate improved its median by at least 3%; the chosen policy was then frozen. Confirmation did not select new winners or change the policy.

## Evidence and limits

The public snapshot contains the 16 original Python source files, contract, environment freeze, protocols, raw timing batches embedded in summaries, selected policy, and reports. It preserves original bytes and experiment paths in those records.

The uploaded review bundle did **not** include original validation receipts, sanitizer logs, profiler traces, or per-case sidecar files referenced by those summaries. Their hashes are recorded, but the omitted files cannot be independently re-audited from this repository. The standard-library verifier checks the available files, source pins, timing arithmetic, and policy consistency; it does not rerun GPU correctness or certify omitted evidence. See [docs/EVIDENCE.md](docs/EVIDENCE.md).

Reported GPU checks covered fixed tolerances, exact residual values, input preservation, fresh outputs, edge cases, and zero-error memcheck runs. Those checks establish tested behavior, not a proof for every finite input. No backward, in-place update, arbitrary-stride support, automatic dispatch to compiled PyTorch, whole-model speedup, or hardware-counter bandwidth result is claimed.

## Project stages

| Day | Work |
|---|---|
| 8 | Define rounding/output contract; check eager reference and compiled pilot |
| 9 | Implement the first fused Triton kernel and smoke checks |
| 10 | Validate the full numerical matrix, edge/API cases, and GPU memory accesses |
| 11 | Measure the initial matrix; inspect launches; tune warps; validate and confirm a frozen policy |
| 12 | Review source/results and prepare publication documentation |

Day labels describe work stages. A clean-checkout GPU reproduction is a separate follow-up; it was not performed during this CPU-only publication review.

[Script map](docs/SCRIPT_MAP.md) · [Engineering notes](docs/ENGINEERING.md) · [Reproduction guide](docs/REPRODUCE.md) · [Evidence scope](docs/EVIDENCE.md)
