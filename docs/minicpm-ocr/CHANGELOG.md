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

## 2026-11-03 — v34, direct FP16 experiment

Opt-in `GGML_SPACEMIT_IME2_FP16_DOWN=1` runs vision FFN-down on IME2.
Weights remain F16, activations convert from F32 to F16, accumulation is F32.
Scope: A100, contiguous K4304/M1152, N divisible by eight. TCM: 337.5 KiB
per worker. M4 fallback guard expands to N988, as tested in v30.

Prior same-binary Q4-text/Q8-vision ABBA: E2E 37.2005 to 26.5480 s.
That is not a Q4/Q4 formal speed result. Q4/Q4 seven-case single-run quality
was text/token-identical in six cases; scanned-table text regressed.
The candidate is experimental and disabled by default.

## 2026-11-04 campaign, started 2026-11-03 — v35

Target: text Q4 / vision Q4. New flag
`GGML_SPACEMIT_IME2_FP16_COMPENSATE=1`, together with the IME2-down flag,
splits F32 activation into FP16 high and residual parts. Two independent
F32 matrix products are added after reduction. The physical N storage is
padded to eight tokens, and stores are bounded by logical N.
TCM allocation is 346624 bytes per worker, including canaries.

This is an experiment awaiting numerical, OCR-quality and ABBA checks.
Both IME2 flags remain off by default. The date names the requested
campaign, not an assertion that tests ran on November 4.
