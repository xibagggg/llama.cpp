# v40 campaign harness

These scripts preserve the exact lab commands used for the frozen runtimes.
They contain task-specific paths and require the local OCR HTTP runner, case
definitions, input images, compatible SpacemiT toolchain and spine/spert SDK.
The case data, SDK and model binaries are not distributed in this repository.
The runner and manifest SHA256 are retained in the private task evidence.
The operator microbenchmark in tests/ocr is deterministic and data-independent.
Runtime directories are immutable; use a new label and destination when rerunning.
PROFILE ON artifacts are diagnostic only. All model speed figures use PROFILE OFF.
