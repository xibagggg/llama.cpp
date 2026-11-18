# MiniCPM OCR：2026-11-18 源码与构建说明

本轮固定 `perf/minicpm`，对应设备 `/home/bianbu/minicpm/minicpm优化/src/llama.cpp`，配套 SDK 为 `/home/bianbu/minicpm/model-zoo-ocr`。完整九项、模型/请求参数、质量边界及整页 ABBA 证据在 SDK 的 `docs/minicpm-benchmark.md` 与 `benchmarks/minicpm_ocr/2026-11-18-mosaic-table/`。这两个仓库的提交分别记录，不能只提交 SDK 就声称 llama 已上传。

整页有效结果是同二进制完整进程均值 **99.235800→75.614324 s，降低23.803%，1.3124×**，尚未达到60 s。收益来自 SDK 原字号文字 mosaic（24逻辑块、24→3物理请求）与显式 nearest-exact 表格512宽，减少 Vision/Prefill 工作；Decode 工作反而增加。不能把它称为本次某个 llama kernel 同工作量提高23.803%。历史 overview 的90.389 s只是 conversion，完整进程为98.905524 s。

## 保留的 llama 实现

| 路径 | 机制与边界 |
| --- | --- |
| `ggml/src/ggml-spacemit/ggml-spacemit.cpp`、`spacemit-opnode.h`、`ocr-q4-pc8.h` | opt-in GELU→PAD→PC8 activation packing及既有bias/residual epilogue；保留图分配、外部输入依赖、packing完成后的barrier及原GELU计算，防止复用buffer先写后读 |
| `ocr-fa-fullkv.h`、`ocr-qk-ime2-v53.h`、`rvv_kernels.cpp` | QK/softmax概率直接采用IME2 blocked panel布局，避免中间重排；GQA K/V packing；保留原attention回退 |
| `ime.cpp`、新增 `ocr-q4-head-prepared.h` | 严格限制Qwen35 `result_output`，M1/K1024/N248100、Q4_0、VLENB128、6worker；在TCM中一次准备原Q8 activation展开和F16修正，复用于N32 panels，不改量化精度 |
| `rvv_kernels.cpp`、新增 `ocr-gdn-tcm.h` | GDN scratch使用每worker现有TCM grant，容量不足回退；保留多token私有state/快照与K1 inplace边界 |
| `ime.cpp`、`ocr-q4-m8.h` | 保留已部署的K4304→N1152 down-projection M8及任意非零K32 block guard；没有重新启用已拒绝K1024扩展分派 |
| 新增 `ocr-op-stats.h` | 默认关闭的按shape/type/op聚合诊断，host准备/退出写出；不把带逐算子profile的时间冒充正式无profile收益 |

开关由冻结profile显式设置：`GGML_SPACEMIT_OCR_GELU_PC8_PACK=1`、`GGML_SPACEMIT_OCR_FA_BLOCKED_SOFTMAX=1`、`GGML_SPACEMIT_DECODE_GQA_KV_PACK=1`、`GGML_SPACEMIT_Q4_HEAD_PREPARED_A=1`、`GGML_SPACEMIT_OCR_GDN_TCM=1`；这些新增路径自身保持opt-in和原路径回退，不能只凭源码存在便假定命中。此前blocked attention独立数值证据为896形状、259375104个FP16概率值/798336个QK值逐位一致，canary完整；完整组合复测及所有输出在SDK历史记录中保留。

本轮删除了失败online attention、head pair/TCM batch、GEMV balance与相关exp-LUT后加接线。扫描当前tracked SpacemiT源码及三个必要新头文件，`OCR_FA_ONLINE/online_enabled/ocr_online_tile/pv16_accumulate_tile/HEAD_PAIR/HEAD_TCM_BATCH/compute2/K1024/ACACHE/Direct32/GDN_ROW_BALANCE/Q4_DISPATCH_DIAGNOSTIC`均无残留引用，`ocr-gemv-balance.h`无include/引用，不进入提交。原有但本profile已关闭的Q4→F16、ALL_K_TCM、FA_X8、normal-exp等开关仍保留原有语义，不能把“仍有旧opt-in入口”说成已在本轮启用。PC8 load threads仍为4。`.bak-*`、`before-*`均为历史备份，不stage；CMake显式列出真正源文件。`ime2_kernels.cpp`当前仅多一个空行，不属于本轮提交。

## AI6、TCM和串行约束

用户已确认AI8会死机；layout与VLM都只用 **AI core8–13、6worker**。实际运行必须保留 `GGML_SPACEMIT_WORKERS=6`、`SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13`、`GGML_SPACEMIT_STRICT_AI_CORES=1`、`GGML_SCHED_REQUIRE_BACKEND=SPACEMIT0`、`GGML_SPACEMIT_SHARED_MODE=PINNED_TCM`、`GGML_SPACEMIT_TCM_OWNERSHIP=1`。`spacemit-tcm-smi -i`有效运行中可看到六块393216-byte TCM busy、两块free；SMI中途样本不等于全程监控。不会以强制CPU fallback掩盖缺失算子。

构建用host CPU0–5、最多`-j4`；构建与推理串行，同设备不能另跑争用AI6的模型。统一锁是 SDK `output/.minicpm-benchmark.lock`。服务请求并发仍为1，threads/threads-batch均6。模型仍为原Q4_0 q4head main、v50 padded-down mmproj，layout仍走系统ORT+EP；当前 `OCR_SPACEMIT_EP=OFF` 不影响SDK系统EP layout，也不把GGUF vision变成ONNX vision。

## 实际构建与当前cache

本轮在源码目录内复用 `build/` 构建，真实成功命令如下；`output/minicpm_ocr/candidates/dspark-build2-20261118/result.json`记录returncode0、295.624 s。这里没有重新声称做过clean build。

```sh
taskset -c 0-5 cmake --build build -j4 --target llama-server test-backend-ops
```

当前 `build/CMakeCache.txt` SHA256 为 `4a813ba3597339f41390ba1ec56e52a9f23888b697284ccab56ea9e12ee063b7`，关键参数如下。旧[2026-11-17构建说明](minicpm-ocr-baseline-20261117.md)是历史配置，不应把其中Ninja/tests=OFF/OCR_EP=ON直接当成本轮cache。

| 配置 | 当前实际值 |
| --- | --- |
| Generator/toolchain | Unix Makefiles；原生 `/usr/bin/gcc`、`/usr/bin/g++` |
| Build type/flags | Release；`-O3 -DNDEBUG`；shared libs=ON |
| GGML_SPACEMIT/CPU/NATIVE/RVV/OPENMP/CPU_REPACK | 全部ON |
| RV_ZFH/ZVFH/ZICBOP/ZIHINTPAUSE | 全部ON；ZVFBFWMA=OFF |
| GGML_CUDA/BLAS/LTO/BACKEND_DL | 全部OFF |
| LLAMA_BUILD_SERVER/TOOLS/TESTS | 全部ON |
| LLAMA_OPENSSL / LLAMA_SERVER_SMT_MTMD | ON / OFF |
| OCR_SPACEMIT_EP | OFF；OCR_EP_DIR/OCR_ORT_DIR均`/usr` |
| SPERT_DIR / SPERT_LIB | `/usr` / `/usr/lib/libspert.so` |
| IME1 / VMADOT_S4 / VPACK / VNPACK checks | 均1，原生工具链实际探测，不能强填成功 |

部署需要完整bin依赖和symlink，不能只复制15-KiB的`llama-server`启动器；其实现位于`libllama-server-impl.so`。部署前先结束自己的server、归档旧部署并持统一锁。build2完整manifest在SDK `output/minicpm_ocr/candidates/dspark-build2-20261118/deployment.json`；相较75秒测量库，本次 `libggml-spacemit.so`已改为 `344a884dafdd8e7dfa300bc38a923276f8d56b567b6058646c256437ebea1ca4`，server实现改为 `c11d035b7be183a3634c06316972409daf384068b6cfacc5dad3090d4603f606`。新提交可能改变build metadata与ELF hash，须保留新指纹，不能套用历史ELF值。

## DSpark：已验证兼容和计时，尚不采用为速度优化

本轮新增严格形状门控的F32矩阵ARGMAX、二维F32 dim1 CONCAT、strided I32 CONT、k=10 F32 TOP_K、I32 GET_ROWS及相应测试，避免draft图在这些点拆到CPU。ARGMAX复用CPU同一函数，保持当前last-index tie及NaN/Inf语义；TOP_K保持CPU当前算法及前两项交换，门控k=10，不宣称更广泛排序/NaN行为。Markov token lookup权重设置NO_REPACK，以保留原量化row布局。I32算子按整数bit处理，避免大token ID经F32损失。

真正的SPACEMIT0对CPU参考定向数值测试证据均在SDK `output/minicpm_ocr/candidates/`，不是只运行“backend skipped”的空测试：

| 证据目录 | 覆盖与结果 |
| --- | --- |
| `dspark-op-validation-20261118-d/` | ARGMAX 9/9、CONCAT 14/14 supported cases通过，含vocab248100、edge值和DSpark concat形状；unsupported形状仍明确回退 |
| `dspark-op-validation-build2-20261118/` | strided I32 CONT blocks1/3/7/19：4/4；TOP_K vocab248100 rows1/15、三种pattern：6/6；I32 GET_ROWS 3/3，含131072个候选ID |
| `dspark-build2-20261118/collector-tests-all.log`等 | speculative计数/嵌套时间与现有schema2 collector回归测试证据；SDK负责汇总 |

数值工具用 `taskset -c 0-5 build/bin/test-backend-ops -b SPACEMIT0 -o <op> -j 1`，AI worker仍是8–13六核；CONT附`-p n_blocks`。仅为允许CPU参考比较而移除测试进程的全图backend强制guard，真实模型运行始终保留guard。此前空skip、CPU参考被guard阻止、I32 CONT缺失的失败目录保留为无效诊断，不合并成通过结果。

`common/speculative.cpp`、`tools/server/server-context.cpp`新增的 `OCR_STAGE/OCR_SPEC`计时区分draft context/proposal、draft generation wall、target verification/plain decode、accepted output steps、拒绝与回滚。draft forward嵌套于generation/process wall，不能重复累加；验证输入tokens含anchor和被拒绝提案，不能冒充输出steps。`OCR_STAGE_TIMING`按环境变量是否存在启用，关闭应unset，设为`0`仍会启用同步/日志。

`dspark-q8-table707-n15-20261118-b/`完成原707×430表格：raw OTSL/final HTML/72格及rowspan/colspan与原输出相同，finish=stop，TCM六块busy，target `n_rs_seq=15`。但HTTP request **42.881 s**，比非spec原表约26 s慢；startup另为13.072 s，不能把局部请求时间称整页E2E。690个提案接受641（92.899%），target decode13.976 s，draft generation19.706 s，server decode总34.424 s，draft成本抵消验证收益。**DSpark Q8目前仅兼容性与瓶颈证据，不是性能采用；默认不加`--spec-type/--model-draft`，75秒有效整页轮未启用spec。**

## 默认关闭的首图启动诊断

随后新增 `OCR_STARTUP_PROBE=1`：`ggml/src/ggml-backend.cpp`在通过backend guard后的首次图提交前输出 `ggml_graph_begin`，`tools/ocr-ep/ocr-ep.cpp`在首次vision EP进入AI guard/session准备前输出 `vision_ep_begin`；均含pid/ppid、CLOCK_MONOTONIC纳秒与backend，GGML另含compute_nodes。每个入口每进程最多一次，以atomic flag避免重复；默认不开启，不加operator callback、backend同步、profile buffer或每图I/O。

SDK对应layout/readiness marker用于判断加载与layout重叠时是否在layout结束前提前启动VLM图，特别是no-warmup候选。必须检查marker存在且backend/compute_nodes与所声称工作相符；空图或host图不能直接当作首次AI计算，缺marker也不能证明没有计算。EP入口是保守的session准备边界，不是算子完成计时。这个诊断增量晚于build2与75秒测量，构建/启动顺序验证另存新label；不据此声称新数值或速度收益。

## 测量版本与本次发布的区别

75秒ABBA测量时 llama HEAD为 `845d3fc07361a0767b04115ec18d07657352f210`，dirty patch SHA256 `595b3d45b1dd81f43ca686a5778587a1ab36cb685bb3f77cab12836e82e48c41`；SDK HEAD为 `eb8f4f855d88ae735a11f404c1e5371b691f0792`，patch `99f18fb6ff77315921daf3f217e537d2be2bdfcbf89deb14c456eedff6da3489`。当时已含基础ARGMAX/CONCAT/NO_REPACK支持，但没有启用spec，也没有后来的I32/TOP_K及spec计时增量。

随后build2源码patch SHA256为 `5118e04c20441113306af361cd690f76d97861dc421479672a42e50dc27c9c79`。两个版本完整源hash/dirty patch和部署hash已分开保存；原75秒关键融合头文件、新增TCM/prepared/op-stats头与当前字节相同。最终发布commit包含本说明和新增兼容/计时，不能冒充测量时HEAD或声称新二进制已复测75秒。SDK证据包 `version.lock.json`明确分离measurement与实际publication字段，实际commit/push及远端SHA由发布步骤填写核验；此文档不预填尚未发生的推送。
