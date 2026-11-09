# RVV copy/transpose assembly contract repair (2026-11-05)

The compiler-contract defect is reproduced and repaired in a server-built candidate. This is a correctness change; no new K3 encoder timing, OCR quality result, or EP parity claim is accepted. The board was accessed for read-only log collection only. The candidate was not deployed to K3 or published.

## Defect and change

The two inline RVV transpose helpers modified v0-v7 and accessed memory without declaring either effect. Their source/destination operands were modified before all input operands were consumed, but lacked early-clobber constraints. `memcpy1d` had the same memory and vector-clobber omissions in its three VLEN branches. These contracts let an optimizing compiler keep live values in registers overwritten by the assembly or reorder surrounding memory accesses incorrectly.

The repair declares the actual vector register groups and memory effects, marks the modified pointers early-clobber, and makes transpose row count a read-only operand. Copy lengths at or below zero return immediately. The arithmetic, transpose loops, positive-length copy instructions, quantization, tensor layout, worker count, synchronization policy, and fusion defaults are unchanged. A compiler `memory` clobber is not a hardware fence.

## Regression evidence

`scripts/ocr/test-rvv-copy-contract.sh` extracts the actual production function bodies without rewriting them, cross-compiles the test at O3 with SpacemiT GCC 14.3, and executes standard RVV instructions under QEMU 10.2.1. The test keeps a vector live across the inline transpose, checks every output and input, and checks destination padding and copy boundary bytes. It includes 16/32-bit transposes, nonzero row padding, row/column tails, and aligned/unaligned copies around 256/1024/2048-byte boundaries.

| RVV VLEN | Before repair | After repair |
|---:|---|---|
| 128 bits | Fails: live vector corrupted | 405 cases pass |
| 256 bits | Fails: live vector corrupted | 405 cases pass |
| 512 bits | Fails: live vector corrupted | 405 cases pass |
| 1024 bits | Fails: live vector corrupted | 405 cases pass |

Every old-binary control first fails on the 16-bit 8-by-1 transpose: transpose output and input preservation pass, but the live vector check fails. Old controls stop at their first failure, so they do not certify later cases. The repaired build passes all 1620 checks. The first harness compilation omitted Zba; that build failed before execution and is excluded. Its log is retained with the experiment evidence.

The complete `ggml-spacemit` shared library also cross-compiles and links using the existing experimental server build and spine/spert 0.6.3. Library SHA256: `6926de9127251428eb83ca7c4cf0a15d02241141e2cba3d7d3f80dc3acb5a589`. The build retains existing FP16 narrowing warnings. The candidate includes the prior dirty fusion/diagnostic experiments; it is not a clean release or a model-level acceptance result.

## Relation to visual drift and EP comparison

A previously uncollected seven-worker replay excluding physical core 13 still changed its fourth formal output: all 258048 values changed, relative L2 0.03572017, max absolute delta 2.22205067. Thus avoiding that core is not a validated repair. The local topology summary now uses the runner's `warmup_exact` field when a saved warmup array is absent, rather than treating a missing file as a mismatch. Original failed model runs remain failed.

The assembly defect is proven independently of that drift. Neither QEMU nor compilation establishes that this repair fixes the K3 failure: it does not reproduce K3 TCM, IME2, multicore scheduling or full model execution. Acceptance still requires uninstrumented model replay, complete embedding comparisons, same-input EP comparison and OCR output checks after board inference is resumed.

The current fast SDK still uses ONNX/EP for layout, MiniCPM vision and PP-OCRv5 CTC, and Q4 llama.cpp for text. Historical H336/W588 vision measurements were ONNX FP16 1.362 s versus GGUF Q4 3.244 s (about 2.38x), with different precision/thread configuration. There is no working PP-DocLayoutV3 GGML/GGUF graph in this branch. This repair does not complete the single-engine SDK.

Reproduce with `RISCV_ROOT_PATH` set to the cross-toolchain and `QEMU_RISCV64` set to QEMU; run `bash scripts/ocr/test-rvv-copy-contract.sh`. Optional `OCR_COPY_SOURCE_FILE` selects an old source snapshot, and `OCR_COPY_TEST_DIR` selects the output directory. These checks report correctness only, not simulated timing as K3 performance.
