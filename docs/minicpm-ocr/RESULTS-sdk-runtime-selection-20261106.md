# 纯 GGML SDK 的配套运行库候选，2026-11-06

父工作区 SDK 新增 `--llama-runtime`，可显式选择 Q4 llama-server 及其配套库，记录实际请求路径。参数检查在取得板端资源之前完成，子进程使用所选目录的库，并清理继承的实验环境。默认仍为 v50，未部署到 K3。

服务器独立交叉构建 llama-server、GGML、MiniCPM 多模态库，以及链接同一组 GGML 的版面／CTC 包装库。候选含 12 个实际 RISC-V ELF、计入符号链接共 28 项，私有动态依赖可在包内解析。27 个接受源码文件与既有构建／测试清单一致，归档哈希经本地复核通过。

服务器的 8 项运行库检查包含真实 x86 Q4 文本请求和异常退出清理，没有执行视觉编码；资源交接回归另含真实 GGML 版面／CTC 冒烟，服务与 PDF 边界为模拟。K3 包没有部署、加载或执行，不能据此认定 RVV／IME2／TCM 验收通过或推算性能。

该阶段补齐入口与配套产物，不新增计算内核或改变默认配置。MiniCPM 文本／视觉继续使用 Q4；版面混合 Q8 和 CTC FP32 是独立模型配置。完整纯 GGML SDK 的 K3／EP 同输入性能、质量与 15／30 秒目标仍待板端验收。未更新飞书、提交或推送。

证据位于父工作区 `results/optimization/20261106-sdk-runtime-selection/`。运行 `python finalize.py` 只复核本地源码、ELF 归档与原始验证结果，不连接设备或启动推理；详见该目录 `REPORT.md`、`FINAL-AUDIT.json` 和 `runtime-k3.tar.gz`。
