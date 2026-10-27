# Evidence included in this snapshot

## Available bytes

This publication package was prepared from `project2-review.zip`, supplied after the final confirmation run. The upload contained the 16 Python source files, original operation contract, environment freeze, Day 11 initial/search/confirmation protocols, summaries, reports, policies and progress records, plus early documentation.

Publication preserves the original Python and contract bytes. All 14 source/contract entries pinned by the archived confirmation protocol match the uploaded files. The recorded environment freeze matches the environment SHA256 in all three stage protocols. Their physical GPU and environment identities agree.

The `results/` copies keep original JSON bytes, including original paths, status flags and hashes. Progress files were omitted from the publication copy because they duplicate completed summary references. The original Day 8 README/specification and intermediate notes are retained under `docs/history/` and should be read as stage-local records, not current result claims.

| Stage | Included file family | Timing data available in summaries |
|---|---|---:|
| Initial | `results/initial/` | 24 × 3 methods × 6 batches = 432 |
| Warp search | `results/warp_tuning/` | 24 × 5 candidates × 5 batches = 600 |
| Confirmation | `results/confirmation/` | 24 × 4 methods × 8 batches = 768 |

The summaries embed the per-case timing records, enabling an independent CPU recomputation of units, medians and reported ratios. Re-serializing their 72 embedded case payloads with the original save format also reproduces the recorded per-case result hashes. No standalone runtime receipts are manufactured by this check. This is stronger than publishing a rounded table alone, but it is not a substitute for all original runtime evidence.

## Omitted original evidence

The review archive did not contain:

- Original Day 8–10 validation receipts or their worker files.
- Original full/smoke selected-policy validation receipts and sidecars.
- Initial and selected-policy Compute Sanitizer logs and saved exit-code files.
- Per-case standalone result files, result sidecars, environment files, and GPU snapshot files referenced by the summaries.
- The original pilot receipt/protocol, launch-inventory receipt, or raw profiler traces.

References/hashes to many of these files remain embedded in the archived protocols/summaries. Their absence is an audit limitation, not a reason to manufacture replacement receipts. `provenance/referenced_files.json` inventories original referenced paths and distinguishes files included elsewhere in this snapshot from files absent in the uploaded review bundle.

Reported successful remote checks were: initial Triton 240 random + 30 edge + 21 rejection checks; compiled baseline 240 random + 32 boundary checks; selected policy 240 random + 30 edge + 21 rejection checks; initial memcheck on 24 smoke cases; selected memcheck on 24 fresh-seed random + 30 edge + 21 rejection checks. Both remote sanitizer console outputs reported zero errors and exit code zero. This publication review did not independently inspect the omitted logs or execute CUDA.

## What the public verifier does

`tools/verify_public_snapshot.py` uses only the Python standard library. It checks the publication hash inventory, parses original Python sources, verifies archived source and environment pins, matches 72 embedded case payloads to their recorded result hashes, recomputes all 1,800 saved timing batches' unit conversions and stage summaries, checks case coverage and rotated method order, and checks selection/confirmation policy consistency.

It does not execute kernels, verify omitted files from their hashes alone, turn the source inventory into an authenticity signature, or establish GPU memory safety. It reports this limitation explicitly. The hashes support internal byte identity, not independent authorship or external certification.

`tools/restore_selected_policy.py` verifies the snapshot and installs only the archived selected policy into its expected runtime location for a demo. It refuses to replace a different local policy and does not create prior receipts or complete a new experiment.

## Fresh reproduction

The original scripts derive their root from their location. A clean checkout can generate a new chain of local validation, memory-check, pilot, search and confirmation records. New hardware/environment/timing values need not match the archived numbers exactly. Once a new study starts, the controllers enforce consistent source, GPU and environment identities within that study.

Copying only archived summaries into live `outputs/` is insufficient: full controllers verify the per-case receipts and evidence that a fresh run must produce. Use the complete ordered guide in `docs/REPRODUCE.md`. A clean-checkout GPU reproduction was not performed as part of publication preparation.
