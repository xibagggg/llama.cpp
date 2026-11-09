# 2026-11-05: EP-style ADD + GELU fusion experiment

The implementation cross-compiles and reduces large ADD+GELU subgraph latency. It is disabled by default. Full-encoder feature stability did not pass, so no encoder-level speedup or EP parity is accepted. No deployed runtime or model was replaced.

## Implementation

`GGML_SPACEMIT_ADD_GELU=1` permits one-consumer contiguous F32 ADD followed by tanh-GELU to write directly to the activation output. Scalar or row bias is supported. Existing use-count/output/view guards remain active. The kernel preserves the existing RVV GELU expression and rounding sequence. It removes intermediate ADD storage traffic and one node barrier; the graph allocator is not changed, so this does not imply lower allocated peak memory. The new opcode is appended to preserve existing opcode values. `GGML_SPACEMIT_ADD_GELU_TRACE=1` reports actual fusion hits. Set switches before creating the graph/process.

## Subgraph ABBA

Base source: `69947e82ae31e4a63df848b6f43949ceab12ffab`. K3, seven workers on 8,9,10,11,13,14,15; host pinned to 0-7; spine/spert 0.6.3. Same binary, OFF-A / ON-A / ON-B / OFF-B. Every arm has one warmup and three measured executions per shape, six measured executions per state. Timing includes graph dispatch, compute and synchronization. The original server remained idle and healthy; it was not replaced.

| F32 shape, width × rows | OFF ms | ON ms | Latency reduction |
|---|---:|---:|---:|
| 1152 × 252 | 0.396501 | 0.314915 | 20.58% |
| 4304 × 252 | 1.351919 | 0.958321 | 29.11% |
| 4304 × 1008 | 4.548697 | 3.230325 | 28.98% |

All four complete output files share SHA256 `f01bf3069f7b16f8d48cf8203cc4e9bff2fd942035cfadef8d076f7df19dc922`. Eleven supported cases cover scalar/row bias, tails, intermediate-output preservation, multiple consumers, full bias, and inplace activation. Inputs are preserved across repeats; the test marks them as OUTPUT to prevent gallocr reuse. A transposed ADD case is skipped because the baseline backend cannot execute that layout. This does not validate unsupported-layout execution.

## Fixed real-image encoder experiment

Actual normalized table pixels from the ONNX bridge were replayed directly, H=336/W=588, output `[1,252,1024]`. HWC input SHA256 is `a19472ff9994dfc3ac046b264caf68852cb16ff3ef624e5dcda3c6efc7007c50`. Visual model SHA256 is `d59bb33cd758ab890e8ea3f697166e87526a90b2e40d0c93d3becfb3b1bd7cae`: Q4 down padded to K4352, `MTMD_OCR_PADDED_DOWN=1`. Text inference, image preprocessing, PDF parsing and layout detection are excluded. This is a pure-vision replay, not a full OCR quality test.

On the current base, OFF formal repeats were not identical: one output changed all 258048 elements, relative L2 0.02854136 and max absolute delta 1.020252. ON's tested formal outputs were exact. The observed means, 3.430211/3.396148 seconds, are not accepted as a speed result.

The same fusion-only patch was independently built on v50 commit `54afbdeaa7709cacdf061171bc107d7bd1257347`, excluding later attention experiments. Four arms, each one warmup and three formal repeats, observed OFF 3.430762 / ON 3.395989 seconds (1.014% less). However, OFF-A1 had one changed formal feature array and ON-B1 had a changed warmup. All ON formal arrays match the common good OFF reference, but the campaign fails the stability gate. The experiment does not prove v50 is generally stable, does not locate a root cause, and does not establish that fusion causes or fixes the drift. These replay flags also differ from earlier fully optimized v50 campaigns.

Each actual encoding reports 27 ADD+GELU fusion hits. Small operator gains therefore contribute only about 1% observed encoder latency here. Model-wide gains cannot be inferred from the 29% subgraph result.

## EP and single-engine assessment

EP's `SpineGraphTransformer::Apply` applies quantization legalization, activation/attention/matmul-bias/transpose fusions and redundant QDQ removal. Its kernels plan TCM tiles and bounded DDR workspace. ggml-spacemit already uses the same hardware family via IME2/RVV/TCM; GGUF is a container, and performance depends on graph and kernels. Fusion alone does not justify a 2× promise.

The local PP-DocLayoutV3 FP16 ONNX graph contains 1258 nodes, including 130 Conv, 164 MatMul and 18 GridSample; it outputs classes, boxes, reading-order logits and masks. No working RT-DETR GGML graph exists in this branch. A single llama/ggml engine requires model loading, graph reconstruction, efficient convolution/sampling kernels and postprocessing, followed by detection/order/mask regression. The fast scanned-PDF experiment also uses PP-OCRv5 CTC ONNX for body/cell text; eliminating ORT requires addressing that model too.

## Decision and reproduction

Keep fusion opt-in. Prioritize reproducing the visual feature issue with per-layer snapshots, then actual-shape QKV/MatMul+bias and D72 QK/PV/softmax work with quality gates. Port layout separately; do not label the full SDK single-engine yet. No full-page stress test ran in this campaign. Original service PID 3064 remained healthy.

Build with `scripts/ocr/build-add-gelu.sh`, setting `RISCV_ROOT_PATH`, `SPERT_DIR`, `OCR_BUILD_DIR`, and optionally `OCR_BUILD_JOBS`. Run `test-add-gelu output.bin` in separate switch-0/1 processes with identical worker pinning, then compare complete files. `bench-clip-fusion MODEL W H INPUT_HWC_F32 OUTPUT_F32 REPEATS` saves every feature array and returns 8 on formal-repeat drift. Output buffers are preallocated to `tokens × dimensions`. Earlier empty-output harness runs were invalid and excluded.

The subsequent Q4 MatMul+bias writeback-epilogue candidate and its separate synthetic/visual acceptance limits are recorded in [RESULTS-ime2-bias-20261105.md](RESULTS-ime2-bias-20261105.md).

Raw logs, arrays and AUDIT.json remain in the task workspace under `results/optimization/20261105-add-gelu-fusion/`; they are not model/source-repository assets.
