# MiniCPM-OCR：虚拟 PAD 与 PC8 down 融合

同一 W588×H336、1008 patches、252 image tokens 裁剪图，重启后独立 ABBA（每进程 1 次完整预热＋3 次正式，每组 n=6）：r8 1.817333 s → r10 PAD-only 1.781696 s，耗时减少 1.96% / 35.64 ms，RSS 532.00→513.91 MiB。12/12 正式完整特征与 r8 正常 PC8-down 参考逐字节一致；四例 OCR 可见文字保持一致，医学表格 15×5，无解析警告。历史漂移根因仍未解决。

`GGML_SPACEMIT_OCR_PC8_PAD_FUSE=1` 将唯一消费者、未输出、连续 F32 的 PAD(4304→4352) 并入 PC8 输入量化，在 workspace 中生成零尾部，读完输入并同步后才写结果。权重 GGUF 仍为 Q4，内部沿用既有 PC8 INT8。独立 profile 节点 514→487、27 个 PAD 全部消除；M1008/7/9/13 与 3 种共享/输出/视图保护模式完成全输出、A→B→A 和输入不可变检查。

`GGML_SPACEMIT_OCR_FA_P_STORE64` 为默认关闭实验。P64 attention 小基准通过，但本轮完整图实验期间发生板端死机，退出/组合初始化的故障阶段未定位，重启后部分文件为零，不能接纳组合或 P64 提速结论。选用配置显式设为 0；没有切换默认服务、提交或推送源码。重启后原临时服务未恢复。

可复核材料：工作区 `results/optimization/20261106-pc8-attention/REPORT-PAD-MORE.md`、`AUDIT-MORE.json`、`OPERATOR-AUDIT-MORE.json`、`QUALITY-AUDIT-MORE.json`、`SOURCE-r10-SHA256.json`、`RUNTIME-r10-SHA256.json` 和 `selected-config-pad-more.sh`。板端候选在 `experiments/pc8-attention-20261106-r10`。完整 PDF/SDK 的 15/30 s 目标未在本轮测试；EP 亦未重测。
