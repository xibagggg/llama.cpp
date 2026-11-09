# Q4 IME2 bias epilogue candidate — 2026-11-05

The opt-in Q4 MatMul+bias candidate now adds the F32 row bias immediately before the IME2 HP kernel's M1/M4 stores. This removes the first prototype's separate RVV output pass. It preserves K accumulation order and accepts only no-zp, full-N32, K256-aligned contiguous Q4_0/F32 shapes with legal single-consumer lifetime guards. Unsupported graphs remain unfused. `GGML_SPACEMIT_MUL_MAT_ADD=1` enables it; the default remains off. The candidate is not deployed as the resident runtime.

Base source: `69947e82ae31e4a63df848b6f43949ceab12ffab`. Isolated server branch/worktree: `opt/ocr-ime2-bias-epilogue-20261105`, `/home/huangjingtao/ocr-ime2-bias-epilogue-20261105`. This build includes existing experimental dependencies from the dirty source tree. The modified no-bias HP kernel is also present when the graph switch is off, so OFF is a same-binary control, not the prior library.

K3 synthetic ABBA, seven AI workers 8,9,10,11,13,14,15 and host 0–7, one warmup plus three timed runs per arm/state pair, six formal runs per state:

| Synthetic K×N×M | OFF ms | ON ms | Latency reduction |
|---|---:|---:|---:|
| 256×128×1 | 0.048081 | 0.056206 | -16.90% |
| 1024×1152×7 | 0.144041 | 0.149624 | -3.88% |
| 4352×1152×252 | 3.712545 | 3.649535 | 1.70% |
| 4352×1152×1008 | 14.541495 | 14.339874 | 1.39% |

Bounded weight/input/bias scales are 1e-4/5e-4/1e-3; K4352 fixtures have 48 zero-padded K elements after logical K4304. Full output files match byte-for-byte across all four arms for each shape. Warmup/formal repeats are exact, inputs unchanged, and each ON execution confirms one fusion. The M252 output SHA256 is `6912a8faebdb03a8e0dcc655e2c52c574b36e0613df5e0c49ef8528b1265952e`, matching the previous post-tile candidate on the same fixture. This is synthetic arithmetic and timing, not trained-model speed or OCR quality. Small cases are noisy and have no demonstrated benefit.

Real H336/W588 crop replay, Q4 padded-down model, [1,252,1024] output: the first OFF arm produced drift in formal r1 versus r0/r2 (all 258048 values changed; relative L2 0.0259749223, max abs 1.1787700653). It exited 8; no ON arm ran. No valid encoder speedup or new OCR quality result is accepted. The issue also appeared in earlier candidates; the root cause is not localized. Keep fusion default-off, locate the first diverging layer, then target D72 QK/PV/softmax and real Q4 packing/TCM work. These results do not establish EP parity or a complete llama-only SDK.

Build: `scripts/ocr/build-add-gelu.sh`, with the documented cross-toolchain and spine/spert 0.6.3. Test: `test-matmul-bias-fusion OUTPUT CASE REPEATS`, cases 0–3 with the above bounded scales. Candidate library SHA256 `24a1dd21a69ac366addc99d209d7c2835c9aa703315ee275f999ef856e8f8e25`. Raw logs, arrays, source snapshot and analysis are workspace evidence under `results/optimization/20261105-matmul-bias-epilogue/`, not source-repository/model assets.

## Follow-up: padding and no-callback worker-count control

After the initial no-callback campaign failed its repeat-stability gate, a single-point callback captured `ffn_down_pad48-0` as F32 `[4352,1008]`. The two captured tensors were byte-identical; every 48-element padding tail in every row was exactly zero. The callback run's warmup and three formal outputs were also byte-identical. This does not attribute or fix the uninstrumented drift because the scheduler callback splits the backend graph into compute ranges and synchronizes each range.

A separate uninstrumented replay of the same normalized H336/W588 input compared seven AI workers (8,9,10,11,13,14,15) with one worker (core 8), each with one warmup and three formal encodes:

| Workers | Formal mean (s) | Warmup/formal outputs exact | Output SHA256 | Max RSS (MiB) |
|---:|---:|---|---|---:|
| 7 | 3.4138 | Yes | `cb6903118a922c205f0faf0655d8ac3d89290ae6437d94de75a7caf65f4e9fc2` | 353.4 |
| 1 | 21.3816 | Yes | `cb6903118a922c205f0faf0655d8ac3d89290ae6437d94de75a7caf65f4e9fc2` | 353.7 |

Seven workers were 6.26x faster than one worker on this crop. Every formal embedding matched across both configurations and the prior stable no-callback reference. This finite replay did not reproduce the earlier intermittent mismatch; it does not override the failed run, prove seven-worker stability, or establish quality equivalence to Q8/FP16. No EP comparison was part of this diagnostic.

The corrected runs and output arrays are archived under `results/optimization/20261105-matmul-bias-epilogue/raw/board-logs/workers-{1,7}`; padding tensor checks are in `padding-checkpoint-analysis.json`. Keep the fusion candidate disabled/default-off until a stable model-level comparison and OCR quality audit pass.
