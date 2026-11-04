# MiniCPM-OCR on SpacemiT K3

This port starts at `spacemit-com/llama.cpp` commit
`855a53bedbc3fdb6f9409e91ca98c2f4f815a182` and preserves its SpacemiT backend.
The default GitHub branch is the integration baseline; experiments use named
branches and opt-in environment flags. Model files and vendor SDK binaries are
not part of this repository.

## What is included

- MiniCPM-OCR conversion and vision preprocessing/merger support.
- Native Q4 IME2 N32 tail handling, physical padding and staging-size fixes.
- Q8/Q6 tails and singleton-matrix dispatch fixes.
- RVV F16 tail-undisturbed accumulation and opt-in four-row TCM cache.
- Acquire fence in pair barriers. This is not an established fix for the
  historical intermittent eight-worker vision-feature drift.
- Existing diagnostics stay behind their compile-time or environment guards.

`Q4/Q4` means text-GGUF Q4_0 and vision-GGUF Q4_0. Original vision
FFN-down matrices remain F16; the opt-in v50 model pads K4304 to K4352 and
quantizes all 27 down matrices to Q4_0. It requires the matching graph adapter
and `MTMD_OCR_PADDED_DOWN=1`. Text output/embedding can remain Q6_K;
activations, softmax and accumulations retain their required precision.
Quantization labels do not imply every operation runs in INT4.

Latest bounded experiment: [v50 results](RESULTS-v50-20261104.md). Formal
same-binary ABBA reduced vision time by about 24%, with one observed missing
bullet in ten OCR cases. v55 attention experiments were slower and showed
unresolved full-model feature differences even in mode 0; use the immutable
v50 tag for the measured speed candidate. Neither version replaces the board
service or establishes a fix for historical feature drift.

## Reproducible build

```bash
export RISCV_ROOT_PATH=/path/to/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR=/path/to/spine-runtime.riscv64.0.6.3
OCR_BUILD_JOBS=16 bash scripts/ocr/build-k3.sh
```

The build writes `OCR_SOURCE_MANIFEST.json` beside `CMakeCache.txt`, recording
the commit, clean/dirty status, tracked source hashes, configuration hash and
binary hashes. A dirty build is explicitly marked and is not a release build.

The four-row cache requires `GGML_SPACEMIT_F16_M4_CACHE=1`. Default execution
keeps the direct path. Use a fixed worker set and one benchmark process.
The tested workaround uses seven AI cores `8,9,10,11,13,14,15`; this does not
establish that CPU12 is faulty or that eight-worker drift has been fixed.

## Update management

1. Record the upstream commit before importing upstream updates.
2. Commit correctness changes separately from performance candidates.
3. Give each candidate a branch, immutable tag, manifest and benchmark record.
4. Compare the same binary with explicit flags when possible. Use warmup and
   ABBA for speed, with OCR/table/formula quality checks alongside timing.
5. Publish unsuccessful experiments as experimental; do not silently promote
   a candidate that changes correct text into incorrect text.
6. Move the default branch only by a normal fast-forward push. Do not rewrite
   shared history or overwrite existing tags. Roll back by a new revert commit
   or by selecting an earlier immutable tag for a deployment.

The running board service is a separate deployment decision. Pushing source
does not replace its binary. See [CHANGELOG.md](CHANGELOG.md) and
[versions.json](versions.json) for version lineage and validation scope.

Current Q4/Q4 results and candidate limitations: [RESULTS-20261104.md](RESULTS-20261104.md).
