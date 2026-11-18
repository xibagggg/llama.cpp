> Device execution update, 2026-11-18: the published `run.py --case all`
> entrypoint has now passed on the actual AI6 device: host 124 + 44 + 8,
> mapped 1326 (948 RVV / 378 fallback), and GDN 48 byte-exact cases.
> Before/after production binding and six-block physical TCM ownership passed.
> Exact commands, logs, SMI and result are in `evidence/publication-device/`.
> The preparation notes below preserve their original historical scope;
> "not executed" in those notes describes preparation time, not this update.
> These numerical tests do not measure page E2E or establish general accuracy.

# MiniCPM standalone regression evidence

Publish this directory as `tests/ocr/minicpm/` in the existing engine `perf/minicpm` branch. These tests do not replace or patch production files. This publication candidate was prepared locally; the new entrypoints have not been executed on the RISC-V device.

The preserved original sources are:

- `mapped/test_mapped_argmax.cpp`: the actual noinline scalar-reference harness, SHA `35a0db291bc89436a63a7f7befb3f9752b048c9c97e3e95ae36f48f8dcfbd731`. Its original device run passed 1,326 independently materialized full-vocabulary cases: 948 RVV cases and 378 original NaN fallbacks. The matching measured header is `mapped/reference/ocr-mapped-argmax-rvv.h`, SHA `9b719ce64aa859f5194d9d83ecc2497b8b57789bf088892ede2afa48c8dc23e2`. Earlier mapped harness failures remain documented by the copied independent review; do not assign the passing result to those failed source versions.
- `mapped/test_mapped_contract.cpp`, its minimal `ggml-impl.h` stub and `contract-headers/`: the exact host metadata/refusal test sources, historically 124 checks. This translation unit explicitly disables RVV. It proves callback/metadata admission, not numerical RVV execution.
- `gdn/frozen48/`: every source, fixture, frozen candidate, original RVV reference, original CPY reference and production TCM ownership helper required by the actual 48-case standalone. `snapshot-bench.cpp` retains SHA `2c2b89ffb50034b72a998b9b2d6358a67b316fdf8dc2dcfc7c1fcffa62b608a1`. The original run checked exact attention/cache bytes, input preservation, padding/guards, repeated same-address requests, real six-worker placement and physical TCM ownership. T16 scalar scratch still uses DDR; this does not establish a new TCM speed benefit.

`SOURCE-PROVENANCE.json` maps each preserved file to its historical source and SHA. `evidence/` holds the original results, commands/flags, numerical logs and independent reviews. `evidence/gdn48/run-standalone-original.py` is the unchanged historical runner for provenance; its old paths are not the supported publication entrypoint. No original/modified engine tree, model, executable, object, `.pyc` or credential is included.

`run.py`, `verify-package.py` and `check-production-binding.py` are derived publication entrypoints. They adjust paths, outputs and locking. Their new device execution has not been verified. They must not inherit the old native PASS merely because the C++ test bytes are unchanged.

Verify the uploaded package and run host checks from the engine repository:

```sh
python3 tests/ocr/minicpm/verify-package.py
python3 tests/ocr/minicpm/run.py --case host --output /tmp/minicpm-host-UNIQUE
```

The host case compiles the preserved mapped 124 metadata checks and GDN 44 admission + 8 bit-pattern copy cases. It launches no AI worker and samples no SMI. Its output must be outside this tracked directory.

The publication runner's host case was executed locally and passed those 124 + 44 + 8 checks; its commands, stdout and result are preserved in `evidence/publication-host/`. All Python entrypoints pass local syntax compilation. `evidence/publication-source-binding.json` records a separate read-only current-device source binding on `perf/minicpm`, with all five header bindings and both GDN/CPY regions matching. None of these local checks is a new native run of the derived runner.

The production binding is a separate read-only check:

```sh
python3 tests/ocr/minicpm/check-production-binding.py --engine-root .
```

`PRODUCTION-BINDINGS.json` ties the mapped/contract and GDN helper headers to the frozen references. Binding permits only the already reviewed CRLF/LF restoration and records actual byte hashes separately. The current production GDN function region must exactly match the frozen candidate, and its CPY region must match the frozen original CPY. The original standalone snapshot header and the later integrated header with the operator-hash-trace guard have different identities; `gdn/integrated-reference/` preserves the guarded version separately. Neither is substituted into the other's historical result.

Native commands are for the parent's serial device schedule only:

```sh
taskset -c 0-5 python3 tests/ocr/minicpm/run.py \
  --case all --engine-root . \
  --sdk-root /home/bianbu/minicpm/model-zoo-ocr \
  --output /home/bianbu/minicpm/model-zoo-ocr/output/minicpm_ocr/candidates/engine-tests-UNIQUE
```

`--case mapped` or `--case gdn` selects one native test. Native execution requires Linux RISC-V, takes the SDK's shared nonblocking benchmark lock, fixes host CPUs0–5 and AI workers6/CPUs8–13, reads machine flags from the production CMake build, and verifies production binding before and after. GDN alone starts one SMI monitor and requires the original physical ownership validation. Builds and tests run sequentially. No option enables eight AI workers. The mapped test preserves the original no-fast-math/no-autovectorization flags and frozen external scalar-reference boundary.

Native compilation also requires the installed production SPERT and TCM development headers/libraries (`spert.hpp`, `spine_tcm.h`, `libspert`, plus the existing Linux device runtime). Those system dependencies are not copied into this source-only test package.

These are numerical and admission regressions. They do not prove arbitrary tensor safety, whole-model equivalence, complete OCR quality or E2E speed. RS0 K1/T16 and history K16 arithmetic are not claimed to be exact equivalents. Full-page Markdown/JSON/image quality and controlled performance measurements remain separate SDK evidence.
