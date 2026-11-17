# MiniCPM OCR K3 基线：2026-11-17

本版本保存已在 Bianbu K3 设备验证的 MiniCPM OCR llama 工作树，以及重建所需的配置。配套 SDK 为 `model-zoo-ocr`。两库使用同名分支 `perf/minicpm` 和同名标签 `minicpm-ocr-baseline-20261117`；复现时须同时检出该标签，并记录两库各自的完整 commit SHA。同名标签关联两库的版本，不表示两库 commit SHA 相同。

SDK 的完整运行方法、质量检查和九项性能指标见其 `docs/minicpm-benchmark.md`。本文记录 llama 构建及源码依据。原始测量及构建产物留在设备的 `/home/bianbu/minicpm/model-zoo-ocr/output/`；模型、二进制和临时备份不随源码提交。

## 已验证源码范围

测量时 llama HEAD 为 `ec1d8101075deb7ab9a172bad0f02d09b9df7761`，另有以下 8 个 tracked 文件修改。本版本应完整保存这些修改，包括开始本轮工作前已存在的调优；只保存新 CONCAT kernel 不能还原已测量工作树。

| 文件 | 内容 | 已验证工作树 SHA256 |
| --- | --- | --- |
| `ggml/src/ggml-spacemit/ggml-spacemit.cpp` | AI worker 默认及最大值为 6；非法显式值终止初始化 | `17054d2cd7fe2cb1a32ee7b8dc435299101d114bedfa0369dc10c7c5e1b65821` |
| `ggml/src/ggml-spacemit/ime.cpp` | 可选 `GGML_SPACEMIT_GEMV_TILE` 覆盖 | `55aaacd40eb50b76e07325e960fe01c81cb47b7730658fdc80345621b883b1be` |
| `ggml/src/ggml-spacemit/ocr-fa-pack-v40.h` | 可选 `GGML_SPACEMIT_OCR_FA_IME_MIN_Q`，默认 988 | `3f605d8aa7f67fac5236b2fc05e4d0c7dcbaf021927b68597ddf94424d169679` |
| `ggml/src/ggml-spacemit/ocr-op-timing.h` | 计时存储容量 8192 → 65536 | `c8eca80a00905ef522e3a6412d017637d38d398a926d566c019a5eb50c283aee` |
| `ggml/src/ggml-spacemit/ocr-worker-events.h` | `concat_segmented` 事件 | `8617a504b8494fb9d58ce155b7b9373c80032b5853dd1188205ba653664e4bce` |
| `ggml/src/ggml-spacemit/rvv_kernels.cpp` | opt-in 单 token segmented CONCAT，形状/stride/type/重叠检查及原路径回退 | `82eb0dd42d8a8832e3d4dbeba464f58a5f46cdd75c8358a5d89d4af56b81f5c4` |
| `tools/ocr-ep/ocr-ep.cpp` | EP 默认及最大线程数为 6；保留原有 SPERT shared-mode 改动 | `ac3f0f6f239051869a6a97ace67539783ba2a95ec350791b35cae6229c6bd249` |
| `tools/server/server-context.cpp` | `OCR_STAGE_TIMING` 开启的 LLM/media 初始化计时 | `0699d6d3bc2c8b36a677f4bc5afead0214b5e02866661090f304dfab4920965d` |

提交前的只读审计逐项比较了 `output/minicpm_ocr/metrics-validation-1.llama.source-hashes.json`：**1937 / 1937 个源码文件匹配，0 缺失，0 差异**。上述 8 个文件均与测量时完全一致。该 manifest SHA256 为 `722a7fbc88197f075115f6d0fac9291f5390d7422c91ec25a7c841dbeda0fe8b`。此结论针对该 manifest 列出的旧源码集合，新增版本文档不在其原始范围内。`git diff --check` 通过。

`ime.cpp` 同时存在 CRLF → LF 转换；忽略行尾差异后整个 8 文件改动为 128 additions、13 deletions，不能把普通 diff 中的大量行尾变化当作大规模算法改写。

`ggml/src/ggml-spacemit/rvv_kernels.cpp.bak-concatvec` 与 `tools/ocr-ep/ocr-ep.cpp.bak-spert` 是未跟踪备份，不在源码 manifest 中；仓库内无文件名引用，CMake 明确列出真正的 `.cpp`。它们不进入本版本，也不参与构建。

## 设备和线程限制

本设备为 `bianbu-spacemitk3picoitx`、`riscv64`，内核 `6.18.3-generic`。用户确认 8 个 AI worker 会导致设备死机。**Layout 和 MiniCPM VLM 都只允许 6 个 AI worker，固定 AI cores `8,9,10,11,12,13`**。不能为性能对照恢复 8 worker，也不能同时运行两套占用这些 AI cores 的推理。

构建只使用 host CPU `0-5`，并行度 `-j4`，不要与推理同时运行。`taskset -c 0-5` 是构建 host CPU affinity，与推理时的 AI worker affinity 是不同配置。

## 原生构建环境

本次实际构建使用设备上的原生 GNU 工具链，没有设置交叉 `CMAKE_TOOLCHAIN_FILE`。GNU 15 的 SpacemiT 指令支持不能用不支持 `xsmtvdotii` 的普通编译器替代。

| 组件 | 实际版本 |
| --- | --- |
| GCC/G++ | `15.2.0-16ubuntu1bb5`，target `riscv64-linux-gnu` |
| CMake | `4.2.3-2ubuntu2` |
| Ninja | `1.13.2-1` |
| ccache | `4.12.3-1` |
| libc6 | `2.43-2ubuntu2bb3` |
| libstdc++6 | `16-20260226-1ubuntu1` |
| spacemit-runtime | `0.7.1` |
| spacemit-onnxruntime | `2.0.8` |

原构建目录为 `/home/bianbu/minicpm/minicpm优化/build/llama`，其 `CMakeCache.txt` SHA256 为 `6e368202b8c711c024c1bcdf29d2f5d6f0cc48eefe12baa3bcd63b21da36716d`。主要配置是 Release `-O3 -DNDEBUG`、shared libs、GGML_SPACEMIT/RVV/OpenMP/CPU_REPACK 开启、CUDA/BLAS/LTO/backend dynamic loading 关闭，server/tools/common 开启。

SPERT 使用 `SPERT_DIR=/usr`、`SPERT_LIB=/usr/lib/libspert.so`。其真实库是 `/usr/lib/libspert.so.0.7.1`，SHA256 `34c85d4da410c4b982902a1578d6bb1a90002e62e5dc7be14a5b158da62460cc`。CMake 检测 `IME1`、`VMADOT_S4`、`VPACK`、`VNPACK` 均为 1，后端定义包含 `RISCV64_SPACEMIT_IME1` 与 `RISCV64_SPACEMIT_IME2`。

实际 CPU flags 含 `-march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d`；SpacemiT 后端 flags 含 `-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb_xsmtvdotii`。不要手动把内部 feature check 缓存强制为成功，应让对应工具链重新检测。

## 从新构建目录重建

以下配置依据已验证构建的 CMakeCache 和生成的 Ninja flags 整理；本次版本保存审计没有另外执行 clean build。路径可调整，但应检出上述标签，并使用同设备兼容的工具链和外部依赖。新构建目录避免覆盖已验证构建。

```sh
set -eu
task_llama_source='/home/bianbu/minicpm/minicpm优化/src/llama.cpp'
task_llama_build='/home/bianbu/minicpm/minicpm优化/build/llama-repro-20261117'

taskset -c 0-5 cmake -S "$task_llama_source" -B "$task_llama_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_SPACEMIT=ON -DSPERT_DIR=/usr \
  -DGGML_NATIVE=ON -DGGML_RVV=ON \
  -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON \
  -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_LTO=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_APP=ON \
  -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_SERVER_SMT_MTMD=OFF \
  -DOCR_SPACEMIT_EP=ON -DOCR_EP_BUILD_PROBE=OFF \
  -DOCR_EP_DIR=/usr -DOCR_ORT_DIR=/usr \
  -DCMAKE_INSTALL_PREFIX='/home/bianbu/minicpm/minicpm优化/out/llama'

taskset -c 0-5 cmake --build "$task_llama_build" --target llama-server -j4
```

`LLAMA_BUILD_MTMD=OFF` 是原 cache 值，不代表 server 不依赖 multimodal library。`llama-server` 目标会构建其依赖；运行部署必须包含 `libllama-server-impl.so`、`libmtmd`、GGML 及 OCR EP 等所需动态库。

仓库已有 `docs/build-riscv64-spacemit.md`、`docs/spacemit-mtmd.md` 是通用/交叉构建说明；本基线使用上述 native 配置，`GGML_CPU_REPACK=ON`、`GGML_OPENMP=ON`，且 `LLAMA_SERVER_SMT_MTMD=OFF`，不能直接套用旧教程的 OFF 值或 SMT vision 运行参数。

## 构建依赖与部署库要分别记录

本构建通过 `/usr/include` 和系统 `/usr/lib` 链接 ORT/EP：

| 构建链接依赖真实路径 | SHA256 |
| --- | --- |
| `/usr/lib/libonnxruntime.so.1.24.2+spacemit.a2` | `381372790da8926a576098e93c125f499f45adc655af5fc94aa7c619a46aba95` |
| `/usr/lib/libspacemit_ep.so.2.0.8` | `0aeadee2b5ed0e8c15a63f8eb676f5ee288ad30542640c304343c6f18562ad2c` |

但已验证 `thirdparty/llama/bin` 内有另一组 ORT/EP 文件；同名 SONAME/版本不能证明内容相同：

| 部署目录中的文件 | SHA256 |
| --- | --- |
| `libonnxruntime.so.1.24.2+spacemit.a2` | `4a674a0a2821a68042ff64ee211d777ea4e9b4f48503a072147f9655df204f11` |
| `libonnxruntime_providers_shared.so` | `59d79f5c77a8b7f9b9567faa709a50bc6da8e03946e746ff68a741eac01283ce` |
| `libspacemit_ep.so.2.0.8-a0` | `00edd31ba9623189e9e151d2ee8139238c3443523e8dafaf354f6157a66cd66d` |

新 clean build 不保证自动生成这些预置的外部 ORT/EP 文件。须另外准备匹配的部署依赖并保存其来源/hash；不能把系统 ORT/EP 替换进来后仍声称部署逐字节一致。SDK 的 layout 进程使用系统 ORT+EP；VLM 基线 vision 使用 GGUF mmproj 的 GGML/SpacemiT0，未开启 `OCR_EP_VISION_MODEL`。链接到 `libocr-ep.so` 不代表实际启用了 ONNX vision。

## 部署与已有二进制证据

已验证部署由原构建目录整个 `bin/` 复制到 SDK 的 `thirdparty/llama/bin/`，保留 symlink 和动态库。构建/部署 manifest 共 **60 项**，逐项一致。60 项是当时原构建树的归档事实，不是 clean `llama-server` 目标保证产出的数量。

```sh
task_sdk='/home/bianbu/minicpm/model-zoo-ocr'
# 先确认没有推理进程占用部署库，并将现有部署另行归档。
# 准备好同版本外部部署依赖后，部署整个 bin/：
cp -a "$task_llama_build/bin/." "$task_sdk/thirdparty/llama/bin/"
```

不要只复制 `llama-server` executable。本次 executable SHA256 为 `276e4a8971b0207a6eebd5cc2ec5f8bde54ef60cb21dedef8d8e4936fe78d724`，`libllama-server-impl.so` 为 `282cd90c9401ca103360df1572cfd5e5ead16db05ebba2ce007adf76ca8301b1`，`libggml-spacemit.so` 为 `a6f8d017239ce13d5c3bab5eb1b9534573aac2d84837415a984212e12ee33e22`。

构建部署证据目录是 SDK 的 `output/codex-20261117/metrics-init-timers-20261117T061435Z/`，含构建命令/日志/结果、部署前备份、build/deploy manifests。后续 commit 会改变 Git build metadata；即使功能源码一致，新构建 ELF hash 也可能改变。应记录新的 commit、工具链、cache、依赖、ELF hashes，再做相同输入/输出和性能验证，不能把旧 ELF hash 当成所有新构建的必然结果。

## 模型与输入

用户提供的模型目录是 `/home/bianbu/.cache/models/ocr/minicpm_ocr`。基线文件及 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `MiniCPM-OCR-1_0-Q4_0-q4head.gguf` | `9b0c4437f0d846a099317d2f86a14284adce5ec601a35bd3404b7d28decb680b` |
| `mmproj-model-q4_0-down-pad4352-all-v50.gguf` | `d59bb33cd758ab890e8ea3f697166e87526a90b2e40d0c93d3becfb3b1bd7cae` |
| `PP-DocLayoutV3.f16.ep.onnx` | `5241af9785736a120c0cd03b798a98863728e9c586ece169bbefd48b4b6c1cca` |
| SDK `data/pages/native-p1.png` | `c46b353f54fb0ef7b95673c31e6e60173566fa4c5621200701454df10b3aac96` |

这些定制模型尚未在本版本材料中提供可验证的公开下载 URL/模型发布版本。**hash 用于核验已有文件，不保证用户能从仓库下载对应模型**；跨设备复现仍需从模型维护方取得上述精确文件及部署运行库。不要用猜测 URL 或其他量化版本替代。模型文件不上传这两个源码仓库。

## 正确性与性能边界

使用 SDK 的 `examples/ocr_file_demo.cpp` 构建的 `ocr_file_demo` 跑单页 `data/pages/native-p1.png`。基线为 `output/minicpm_ocr/metrics-validation-1`；其 `.profile.json` 是本次参数和显式环境配置来源，不能仅按 kernel 默认值还原测试。Layout 和 VLM 均为 6 workers/core 8–13，VLM 单槽串行、`temperature=0`、`seed=42`、`cache_prompt=false`、`OCR_STAGE_TIMING=1`。

结果为 29 个 layout blocks，24 个 VLM 请求，5 个 `skipped_no_vlm`，0 个失败。`recognize_ignored_labels=false` 跳过默认 Markdown 忽略标签对应的识别请求，但 JSON 保留这些区域。输出 Markdown 与图片相对上一份最终结果逐字节一致。质量一致性是相对本输入及已保存输出的回归结论，不是所有文档准确率的保证；完整输出保存在 `output/minicpm_ocr/metrics-validation-1/native-p1/`，便于人工查看表格、公式和图片。

| 指标 | metrics-validation-1 | 解释 |
| --- | ---: | --- |
| Prefill load | 5199.585 ms | 共享 LLM 模型初始化 |
| Decode load | 同上，共享初始化 | 与 Prefill load 不能相加 |
| Vision load | 1721.660 ms | mmproj/media 初始化 |
| TTFT | 3531.277 ms | 首个有效 crop 服务端 TTFT，不是整页 wall |
| Vision | 54489.024 ms | 24 请求的 vision 阶段累计 |
| Prefill | 23902.560 ms | 累计 prompt eval |
| Decode | 36400.118 ms | 累计服务端 decode 区间 |
| TPOT | 28.526738 ms/token | decode 区间 / 1276 decode steps |
| E2E | 121308 ms | SDK 转换区间，不含独立冷启动 load |

冷进程 wall 为 129819.676 ms，layout 5584 ms，VLM 115542 ms；1300 completion tokens、1276 decode steps、6959 prompt tokens。首 token 的计数边界使 completion tokens 与 decode steps 不同；TPOT 不使用 completion tokens 作为分母。纯 decode forward 计时为 35152.698 ms，不能与上述 decode 区间混写。

这些数值是一次已完成的正确性回归和性能观测，不是多次统计结果。再次测量要保留独立 run 名称、profile、日志、输出、metrics/provenance 和 source/runtime hashes。相同 source tag 仍需核对模型、运行库、环境和参数，尤其不能改变 worker 数或推理并发后继续直接比较。
