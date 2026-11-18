# OCR merged verification 2026-11-18

All runs: native-p1, six AI workers8-13; host0-5; builds and inference serial. No page-cache flush. Two full-page measurements per model/branch. MiniCPM interleaved ABBA; Paddle baseline and fixed candidate measured in separate pairs after repairing model-specific layout ordering. This is one-page relative regression coverage, not corpus accuracy or proof about every model.

| Metric (seconds; TPOT ms/token) | Mini baseline | Mini merged | Paddle baseline | Paddle merged |
|---|---:|---:|---:|---:|
| prefill_model_load | 4.837283 | 4.955908 | 3.363307 | 3.130359 |
| decode_model_load | 4.837283 | 4.955908 | 3.363307 | 3.130359 |
| vision_model_load | 1.852875 | 1.797120 | 1.685133 | 1.786735 |
| ttft | 1.791735 | 1.770817 | 2.291346 | 2.240462 |
| vision | 9.059778 | 8.986669 | 10.883465 | 11.030115 |
| prefill | 7.194130 | 6.548896 | 4.735196 | 4.755997 |
| decode | 28.705732 | 28.974243 | 24.960268 | 25.148694 |
| tpot | 20.358676 | 20.549108 | 17.714881 | 17.848612 |
| e2e | 53.282196 | 52.936329 | 47.381020 | 47.726598 |

Prefill/Decode load share initialization; do not add twice. MiniCPM TTFT is server first token; Paddle TTFT is client first nonempty streamed content. Compare only within model.

MiniCPM full conversion times: baseline53.427/51.597s, merged52.422/51.866s. Paddle process E2E: baseline47.452199/47.309841s, merged47.649190/47.804006s. Paddle merged mean is about0.35s/0.73% slower in this limited pair; do not claim zero measurable slowdown or a speedup. Request count and output work restored: Paddle7 requests/1416tokens/1409steps; MiniCPM3 requests. Both merged repeats pass complete Markdown comparison (only image-reference targets replaced by referenced image SHA256),72td/15tr and retained PNG hashes,0 failed/truncated blocks. MiniCPM Markdown is byte-identical.

SDK CTest18/18 PASS. Layout regression explicitly verifies model order when repair is disabled and existing MiniCPM heading repair when enabled. Initial Paddle8-request ordering regression was rejected and fixed via LayoutOptions.repair_heading_order, disabled only by Paddle adapter; generic/MiniCPM remain unchanged. Earlier measurements are retained diagnostics, not accepted results.

Raw evidence: /home/bianbu/ocr-merged/evidence and each baseline/merged project output directory. Published JSON copies live beside this document in docs/ocr-merged-evidence. Frozen input revisions and upstream design notes: ocr-merged-integration-20261118.md. No upstream runtime push.
