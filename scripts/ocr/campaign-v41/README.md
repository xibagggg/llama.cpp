# v41 lab campaign harness

The commands retain exact lab paths and immutable runtime/result names. Copy
archived files into the original OCR workspace scripts directory before use;
relative imports and project-root resolution assume that location. Use new
names when rerunning. They need the local OCR benchmark module, private images,
case definitions, actual captured tensors, compatible toolchain and spine/spert.
These dependencies and model/SDK binaries are not distributed here. The real
FFN-down replay in tests/ocr consumes the private v36 capture directory; it is
not a data-independent public benchmark. Reports contain bounded numeric data
only. PROFILE ON and coarse internal probes are diagnostic, not speed figures.
