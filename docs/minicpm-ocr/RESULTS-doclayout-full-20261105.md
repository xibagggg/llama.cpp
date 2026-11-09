# PP-DocLayoutV3 native GGML follow-up — 2026-11-05

The full FP32 PP-DocLayoutV3 graph now executes in the project's GGML CPU model runner, without ONNX Runtime. This is an additional model runner and C library in the llama.cpp repository, not support for passing the layout GGUF to an unmodified upstream llama-server. MiniCPM vision/text retain their Q4 GGUF path; PP-OCRv5 CTC has a separate native GGML implementation.

The static layout export preserves all four outputs: class logits, boxes, reading-order logits and masks. The 1408-node source graph contains 130 Conv, 137 Gemm, 27 MatMul and 18 GridSample nodes. Numerical comparison against ONNX CPU passed on a paper-table crop and a 300-DPI scanned paper page. The latter retains 29 regions with equal labels, rounded boxes, polygons and all 300 query reading-order ranks. Repeated GGML requests and CLI/C API outputs are byte-identical. Persistent SDK validation verifies input preservation, error recovery and shutdown, and confirms that ORT was not imported.

Two guarded optimizations expose GGML native FP32 elementwise operations (130 hits) and native compact copies for supported layouts (55 hits). The C API options are a bitmask: bit 0 for elementwise operations; bit 1 for compact copies. Other layouts retain the original custom path.

On the shared x86 build server, a complete A/B/B/A campaign measured 12.4257 to 6.9923 seconds for combined optimization, 43.73% less latency. Each arm excludes one warmup and averages three formal runs; each state has six formal runs. All four full output tensors are byte-identical across states and repeats. The separate compact-copy ablation measured 8.5679 to 8.2142 seconds, but overlapping ranges and load drift do not establish a reliable independent gain. The bounded combined command did not complete the second campaign; only that campaign was rerun and the incomplete samples were excluded and retained in the evidence archive.

These are x86 CPU observations. K3 executables and libraries cross-compile, but have not been deployed or executed. No board service, Feishu document, commit or push was changed in this follow-up. The new layout/CTC graphs are FP32 correctness references and explicitly use GGML CPU; IME2/TCM acceleration, EP parity and full SDK latency/stability remain unverified. MiniCPM's RVV contract repair still needs K3 feature-repeat validation.

Build entry points:

- `scripts/ocr/build-doclayout-ggml.sh`: native server CLI and shared library.
- `scripts/ocr/build-doclayout-ggml-k3.sh`: RISC-V cross-build, no deployment.
- `scripts/ocr/convert-doclayout-gguf.py`: static ONNX export to GGUF; ONNX is a conversion dependency, not an inference dependency.

Full arrays, overlays, build hashes, SDK checks and raw ABBA samples remain outside Git in the workspace evidence directory `results/optimization/20261105-doclayout-ggml/`, with the local `REPORT-FULL.md` and `BUILD-FULL-MANIFEST.json`.
