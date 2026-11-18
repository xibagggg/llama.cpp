# OvisOCR2 integration (2026-11-18)

Merged into perf/ocr-merged without textual conflicts:
- SDK source feat/ovisocr2-k3: 88da558ee2d2888f794148651640a36f8d46bbc7
- Engine source feat/ovisocr2-gguf-k3: 7587d8a9ea36f3718ee698055337c80030306aef
- Previous SDK target: 7e7fef5719bc6c2ab51904937e5376ce21d14705
- Previous engine target: 87bede1a0da8ddf11784134d07ea8bc9949bf1c2

Engine full build passed. SDK build with OCR_WITH_OVISOCR2=ON passed.
SDK CTest: 18/18 passed. Updated preset count from 7 to 9 and added
OvisOCR2 preset/provider/thread and backend-name round-trip assertions.
Builds ran serially with taskset -c 0-5 and -j4.

OvisOCR2 SDK is backend-local. Engine adds optional still-image patch support,
Qwen3VL preprocessing/conversion changes and a scale-bias correctness fix.
Existing Paddle/MiniCPM profiles and document pipeline policies are retained.
This is a source/build integration, not proof of inference equivalence.
No model inference or performance benchmark was run. OvisOCR2 model is not
available; quality and performance comparisons for all affected models remain
pending at the user's request.
