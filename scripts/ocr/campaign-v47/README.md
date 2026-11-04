# v44-v47 laboratory evidence harness

Experimental and opt-in. Copy scripts into the original OCR workspace scripts
directory before running; imports and ROOT resolution expect that layout. Use
new result/runtime names and a new explicit deadline for reruns. The original
private benchmark module/cases/images, v43 quality reference, compatible cross
toolchain and spine/spert are required and not distributed. Real down capture
is needed for its replay; model/SDK/captures/pages/raw OCR are excluded here.

Final selected path: complete v43 combination plus DOWN_VECTOR=5, FA_DIRECT=2,
CONCAT_TCM=4; FA_GROUP remains unset. v47 short/partial CONCAT falls back. The
same-binary ABBA measures their combination, not independent page attribution.
The soak validates exact outputs/tokens/saved final-slice features per request,
records three-second telemetry, stops starting requests at its deadline and
restores the original service. PROFILE ON is diagnostic only. Historical
eight-worker drift remains unresolved; retain seven-worker scheduling.
