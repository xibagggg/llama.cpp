# Paddle serial-v3 paired QK admission

Default-off GGML_SPACEMIT_OCR_FA_SERIAL_P2_PAIR_QK=1 reuses one K tile copy across the original high/low QK calls. It retains original arithmetic and existing v3 admission/fallback. Requires SERIAL_P2=1 or2. Disable PAIR_QK to restore original v3; no model change, TCM growth or added workers.

Build the server, then compile fixtures using the exact rvv_kernels.cpp compile flags from build-paddle/compile_commands.json, include ggml/src/ggml-spacemit and SPERT includes, link the same build libraries and SPERT. Root retained exact commands and binary hashes in vision9-20261118/pairqk-freeze1/build-manifest.json.

On K3 set SPERT_BACKEND=spacemit-k3, SPERT_SHARED_MODE=TCM, SPERT_RT_PRIO=0; WORKERS=6, ALLOW_EIGHT_WORKERS=0 and preferred8..13. Run test-qk-pair-a1-stream SCATTER PIPELINE for00/01/10/11. Each runs1536 exact original-kernel comparisons on lane0, all six workers require actual CPU8..13/VLEN128/TCM393216. Four processes total6144 cases, including tails/guards/input immutability. This is numerical verification, not a performance benchmark.

Then run python3 run-graph-a1.py --binary ABS/test-paddle-attention-v3-a1 --library-dir ABS/build-paddle/bin --output FRESH_DIR. It requires92 graph executions each arm, full F32 bits equal to frozen v3 and exact multiset1095552 paired tile hits across32 valid graph executions. Original v3 FP64/no-worse and fallback thresholds remain inherited; no tolerance change.

Root raw admission1 had an incorrect aggregate6144-per-process count guard. Original log had numerical PASS1536 and placement PASS. Kept as FAIL, fixed driver guard only; admission2 all four processes and184 graph executions PASS. Correct factors are6row choices x64count choices x4variants. Full untraced page ABBA and quality are independent records, not inferred from fixtures.
