# Bounded v61 attention campaign

Use the immutable minicpm-ocr-k3-v61-experimental tag for the measured runtime.
Documentation is a later commit; do not substitute its commit for the runtime
manifest. Build with scripts/ocr/build-k3.sh and the recorded SpacemiT SDK.
The existing v50 padded Q4-down GGUF and MTMD_OCR_PADDED_DOWN=1 are required.
GGML_SPACEMIT_OCR_FA_COMPUTE=0/3 is the only formal ABBA difference.

Copy board campaign runners into the existing project's scripts/ directory
so project-root and benchmark.py imports resolve. Existing models, images,
benchmark support and ground truth are task assets and are not included.
Build scripts record original lab paths; adapt those paths for another lab.
Boundary benchmark and rejected isolated PV prototype sources are archived
under tests/ocr. They must be built against the measured runtime objects.
Do not regenerate runtimes with an existing immutable folder name.

PV unroll is a rejected experiment. Kernel-only gains are not model gains.
Ten selected outputs match v50, but last-slice features are not bitwise equal
across compute modes. Formal repeats are also not bitwise stable within both
OFF and ON; strict acceptance failed. summarize_attention_v61_observed.py
records descriptive timings with passed=false; audit_attention_v61.py remains
strict and still fails. All numeric details and limits are in the result record.
