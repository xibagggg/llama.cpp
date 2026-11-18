# OvisOCR2 merge validation 2026-11-18

Original branch vs merged branch, same board/native-p1 input and exact archive weights. Six AI workers8-13, host0-5, context16384, batch/ubatch1024, one slot, warmup enabled, 64-576 image tokens. No cache flush; fresh processes, not cold disk. Serial builds and inference. Run order merged1, baseline1, merged2, baseline2.

| Metric | Baseline mean | Merged mean |
|---|---:|---:|
| process_e2e_s | 84.284949 | 84.498100 |
| conversion_s | 77.443500 | 78.720000 |
| decode_s | 48.365027 | 49.340484 |
| tpot_ms | 30.669009 | 31.287561 |
| server_prompt_including_vision_s | 22.702687 | 22.982450 |
| startup_connection_s | 6.237500 | 5.231000 |
| layout_s | 5.649720 | 5.655862 |

All four full Markdown files are byte-identical; retained image hashes identical;29 blocks,24 requests,1601 tokens,1577 generated steps,zero failures/truncations per run. This proves relative output preservation on one page, not corpus accuracy or human ground truth. Table/heading/formula content is covered by full-byte comparison.

Nine-metric JSON explicitly leaves model load, Vision, pure Prefill and TTFT null: native Ovis instrumentation does not expose them. prompt_ms includes Vision and must not be presented as pure Prefill. TPOT uses total decode / sum(output_tokens-1). Screenshot78.740s lacks original timing provenance; current process E2E includes startup/exit and must not be equated to screenshot. Report both conversion and process wall, without claiming a screenshot speedup.

The archive README asks for default warmup; both comparison profiles remove repository --no-warmup. Native model files are at /home/bianbu/.cache/model/ocr/ovisocr2. Layout is supplied explicitly from the existing verified EP file. No model or inference source changes. Prior Paddle/MiniCPM model regression tests were not rerun in this Ovis-only validation.

Raw outputs/logs/commands/profiles/build logs: /home/bianbu/ocr-merged/evidence/ovisocr2-validation. Source versions and model/input/layout hashes plus per-run evidence: ovisocr2-validation/comparison.json.

Measured merged process mean is 0.213151s (0.253%) slower; conversion mean is 1.2765s (1.648%) slower and decode mean is 0.9754565s (2.017%) slower. Limited two-run samples do not prove zero performance regression or a statistically stable slowdown. Preserve these observations; no performance fix is claimed.
