# v42 lab campaign harness

These scripts preserve exact lab paths and immutable result/runtime names.
Copy them back into the original OCR workspace scripts directory before use;
relative imports and project-root resolution assume that layout. Use new names
for reruns. Private case images, the v36 real-tensor capture, original OCR
benchmark modules, compatible toolchain and spine/spert are required. These
dependencies, models and SDK binaries are not distributed here.

The standalone initial replay runtime combines the new micro executable with
v41 libraries; the production and profile runtimes are complete clean rebuilds
of the recorded tested commit. All three modes pass exact replay; only paired
has complete OCR and formal performance validation. Small-N replay tests tails
directly; production dispatch remains guarded to N>=988. PROFILE ON is a
diagnostic and must not be compared by subtraction with PROFILE OFF.
