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

### v35b — scale residual during FP16 packing

The initial actual-forward microbenchmark at N1 was finite and tail-safe,
but relative L2 versus M4 was 9.04e-6, above the preset 2e-6 threshold.
Investigate by multiplying activation residual by 4096 before FP16 packing,
then scale its product back. This is an exact power-of-two adjustment;
it does not establish the IME subnormal mode as the cause.

## 2026-11-03 — v36-v40, priority optimization campaign

Actual down tensor capture and selected-layer dispatch; normalized tiny FP16 weight/activation handling; actual dynamic-quantization boundary capture; complete operator profile flushing; exact D72 eight-query attention reuse with mask support; transient cooperative K/V packing. All new performance paths are opt-in.

See RESULTS-20261104.md and benchmark-results-v40.json for failed all-layer quality experiments, seven-case gates and formal same-binary ABBA. No default deployment promotion and no eight-worker drift claim.

## 2026-11-03 — v41 activation TCM tiling

Optional exact M4 activation tiles 8/16 with shape/scratch guards and diagnostic coarse phase probes. Real-input frozen-reference replay, seven OCR checks, same-binary incremental ABBA, operator profile and bounded numeric results. All new flags remain unset by default.

## 2026-11-03 — v42 paired down scheduling

Optional 2x2 weight reuse over v41 TCM16 with unchanged per-output arithmetic. Frozen-reference replay across nine sizes including odd tails, seven OCR quality cases, incremental same-binary ABBA and full operator profile. Added bounded timestamp recovery for one interleaved quality-request log, preserving original evidence. New scheduling flag remains unset by default.

## 2026-11-04 — v43 compact down and attention initialization

Optional compact20 paired down with two weight scratch rows; skip only overwritten attention initialization, retaining tail padding and fallback. Exact down replay, poison-scratch attention replay, seven OCR cases, same-binary combined ABBA and full operator diagnostic. Both new switches remain unset by default.

## 2026-11-04 — v44-v47 long campaign

Opt-in aligned down scratch/direct packed V reads and bounded text CONCAT TCM transpose. Measured short/partial-write regressions cause explicit original-path fallback. Wider-query and K-direct negative results retained.5,184 exact formal micro runs, full OCR and four table GT cases, final combined three-case ABBA, full operator profile and deadline-bound mixed-document soak. All new switches remain unset by default; permanent deployment unchanged.
