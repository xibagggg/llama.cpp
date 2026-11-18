# Paddle large-row Attention admission

These are six-worker K3 numerical fixtures for the default-off
`GGML_SPACEMIT_OCR_FA_SERIAL_P2=1` route. They retain the old-shape and invalid
input F32 equality requirements and the original valid-large-shape FP64
relative error / no-worse thresholds. They are not performance benchmarks.

Build the two `*-stream*.cpp` programs with this repository's ggml-spacemit
RVV/IME2 compile flags, the backend include directory and the configured
SPERT include/library directories. Build the two public ggml graph fixtures
against the same build's ggml, ggml-spacemit, ggml-cpu and ggml-base libraries.
The CPU library supplies the existing backend reference, and is not modified.

Under the board benchmark lock, with six workers on preferred AI cores
8,9,10,11,12,13 and SPERT_SHARED_MODE=TCM, run query-stream (360 cases),
softmax-stream (525 cases), then the graph fixture with `--write-reference
NEW_FILE` and `--compare-reference FILE` (92 executions each). Finally run
`run-exclusion-v3.py --binary EXCLUSION_BINARY --library-dir BUILD/bin
--output NEW_DIRECTORY` (11 gates, 33 processes, 264 graphs). Never run the
RVV body on an ordinary host CPU. Existing references and output directories
are refused. Keep the original thresholds and preserve failure evidence.

The SDK baseline document sections 50 onward record model hashes, commands,
numeric/quality admission and independent untraced full-page ABBA timings.
