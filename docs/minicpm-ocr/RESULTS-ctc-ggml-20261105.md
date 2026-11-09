# PP-OCRv5 CTC in native GGML (2026-11-05)

PP-OCRv5 CTC now has a shape-specialized FP32 GGUF exporter and a native GGML CPU runtime, with a persistent C API. This removes ONNX Runtime from this recognition component. It is a correctness reference, not an IME2/INT8 implementation or a complete llama-only document SDK. No K3 inference, deployment, commit or push was performed for this candidate.

`scripts/ocr/convert-ppocr-ctc.py MODEL.onnx MODEL.gguf --batch 8 --width 512` exports 217 FP32 tensors and 260 runtime nodes. Only 16 integer shape/index nodes are folded. Convolution uses FP32 im2col plus MatMul; depthwise convolution, activation, pooling, normalization, attention and Softmax execute in GGML. Shape metadata is refreshed before concrete inference. Temporary rank-five QKV transforms are metadata views, materialized at rank four or below.

`tools/ocr-ggml/ctc-ggml.cpp` builds a standalone CLI or, with `OCR_CTC_LIBRARY`, `libocr-ctc-ggml.so`. The public C header provides handle creation, shape queries, execution and destruction; errors use caller-owned buffers and execution is serialized per handle. Inputs retain their allocation across calls, and each request rewrites the input. This fixes an early harness failure caused by input/scratch aliasing. Preserve that failed evidence separately from accepted results.

Full ONNX CPU probability references with graph optimization disabled, using actual benchmark table crops:

| Batch / width | Output shape | Maximum absolute probability error | Relative L2 | All timestep argmax and decoded text |
|---|---|---:|---:|---|
| 1 / 128 | 1 x 16 x 18385 | 1.073e-6 | 5.091e-7 | Equal |
| 1 / 512 | 1 x 64 x 18385 | 2.384e-6 | 3.980e-7 | Equal |
| 2 / 256 | 2 x 32 x 18385 | 2.682e-6 | 8.332e-7 | Equal |
| 8 / 512 | 8 x 64 x 18385 | 7.749e-6 | 8.948e-7 | Equal |

Every case also passes finite/probability-normalization checks, preserved-input checks and byte-exact repeated outputs. The persistent SDK adapter in the parent workspace shares preprocessing and CTC decoding with its ONNX adapter. Three shapes pass reordered inputs, partial batches, repeated requests, invalid-size rejection without output writes and recovery, width rejection and closed-handle checks. The SDK process never imports ONNX Runtime. The limited English table fixture is not a Chinese OCR or full-page accuracy benchmark.

Build the x86 reference with `bash scripts/ocr/build-ctc-ggml.sh`. Build against existing K3 GGML libraries with `RISCV_ROOT_PATH`, `SPERT_DIR` and `OCR_CTC_K3_BUILD_DIR` set, then `bash scripts/ocr/build-ctc-ggml-k3.sh`. Both the RISC-V CLI and shared library link successfully. They explicitly initialize the GGML CPU backend; linking the SpacemiT backend does not imply IME2 scheduling.

Candidate SHA256: x86 shared library `9ecf0c16e5ec4d5cfc38a72b1750ac2270d49d353a16db3126bf23e512e99a74`; RISC-V shared library `aea533a3b75ba649aecea1abfbf71ce8ad6d7c01a2efb3024eee104635cc3dd8`; RISC-V CLI `d0131b1d6a35c653777c27f917ece18ff5e05952d2db209ce6e684dcb752510a`. Base source is `69947e82ae31e4a63df848b6f43949ceab12ffab`, plus the existing dirty experiments. Complete raw arrays, server logs, source/binary manifests and audits are workspace evidence under `results/optimization/20261105-ctc-ggml/`.

Remaining: width-bucket/dynamic-shape handling, quantized kernels and IME2 routing, K3 stability and performance acceptance, PP-DocLayoutV3 full graph conversion, and end-to-end EP comparison. Historical vision measurements remain FP16 ONNX 1.362 s versus Q4 GGUF 3.244 s on the same H336/W588 crop with different precision/threads; this work adds no new board comparison and does not validate the document latency targets.
