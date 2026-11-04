# v43 lab campaign harness

Copy these scripts into the original OCR workspace scripts directory before use;
imports and relative project-root resolution expect that layout. Use new result
and runtime names for reruns. Private images/cases, original OCR benchmark modules,
the v36 real layer capture, compatible toolchain and spine/spert are needed and
not distributed here. No models, SDKs, pages or captured tensors are committed.

Production and profile runtimes are complete clean builds of the tested source.
The two changes are microbenchmarked independently; full-request ABBA measures
the selected combination. Attention micro timing includes prepare and scratch
poisoning; PROFILE ON is diagnostic, not a production speed benchmark.
