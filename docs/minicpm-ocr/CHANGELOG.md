# MiniCPM-OCR K3 change log

## 2026-11-03 — source-management baseline

Imported the existing OCR port, Q4/Q8/Q6 tail and singleton corrections,
F16 tail handling, guarded diagnostics and opt-in four-row TCM cache from
the working source, preserving the SpacemiT backend. Normalized only line
terminators in those files to avoid a whole-file diff from CR-CR-LF.

Historical six-case comparisons use a separate v8 candidate with seven
workers and M4 enabled. Their Q4/Q4 mean E2E is 43.93 s versus a historical
single-run 269.55 s. This is a cumulative comparison, not a causal estimate
for any individual patch. It excludes the later IME2 FP16 experiment.

The original working tree and local runtime directories remain preserved.
This source snapshot has to pass its own cross-build before publication.
