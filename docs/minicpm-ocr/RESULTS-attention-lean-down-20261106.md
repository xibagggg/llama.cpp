# Q4 文件内的 attention 布局与 PC8 down M8 优化

K3 同一 W588×H336 医学表格裁剪，旧 r4、新 attention、再加入 PC8 down M8 三种配置按 A/B/C/C/B/A 顺序运行。每进程预热一次、正式三次，每配置正式 n=6：

| 配置 | 视觉编码 s | RSS MiB | 较本轮旧版减少 |
|---|---:|---:|---:|
| 旧 r4 | 2.009418 | 476.57 | — |
| attention 优化、原 down | 1.881652 | 474.68 | 6.36% |
| attention＋PC8 down M8 | 1.802396 | 532.14 | 10.30% |

文本和视觉 GGUF 仍为 Q4。视觉内核按用户授权使用 INT8 PC8：原有 QKV/O、up 的逐通道/逐 token INT8 扩展到 K4352 down，保持完整 K INT32 累加及 F32 epilogue。M8 让两组 M4 共享权重加载/解包，TCM 容量不足时保留 M4。两层 K4608 merger 不在 PC8 选择范围内。

attention 新增默认关闭的开关：

```sh
GGML_SPACEMIT_OCR_FA_LEAN_PREPARE=1
GGML_SPACEMIT_OCR_FA_QK_SCATTER=1
GGML_SPACEMIT_OCR_FA_BATCH_CHECK=1
GGML_SPACEMIT_OCR_FA_FIXED_INDEX=1
GGML_SPACEMIT_OCR_Q4_PC8=2
GGML_SPACEMIT_OCR_Q4_PC8_DOWN_M8=1
```

沿用 `FA_FULL_KV=1`、`FA_FULL_PACK=1`、`FA_Q32=1`、`FA_SOFTMAX_PACK=1`、`Q4_M8=1`、`Q4_PC8_M8=1`、`Q4_PC8_MULTI_N=1`、`Q4_F16=0`，7 worker 绑定 8,9,10,11,13,14,15，host 0–7。独立脚本 `selected-config-next.sh` 保存完整环境。

full-KV 准备不再同时构造备用 K/V。若 full-KV 失败，旧布局从原张量局部重建，并保留原 IME2 QK 分派与清零。QK 结果由连续 8×8 读取后 scatter，替换逐行 gather；softmax 保持原求和分组，合并完整掩码检查并复用对齐 panel 索引。`FA_FUSED_QUERY` 正确性通过但未获得净收益，最终关闭。

r7 独立节点诊断：attention 632.384→489.952 ms，down 374.444→295.989 ms；514 个正式节点，每节点 7 worker，无记录丢失。算子时间为关键窗口；阶段诊断另用 worker 时间和÷7，不能混加。

N988/1008/1120 的单/双遍 attention、Q/K/V 有限超范围回退、A→B→A 与输入不可变检查通过。down M1008/7/9/13 的 PC8 M4/M8 完整输出一致。r5 的回退差异已由 r6 修复，失败数据保留。

最终三配置共 18 次正式输出在配置内一致。attention 单独优化保持原特征；新增 down 重量化后形成另一正常特征，relative L2 约 0.120131，不能解读为准确率下降 12%。r7 两种候选的四个真实 OCR 小样本可见正文均与此前一致；r8 最终医学表格复查为 15 行×5 列，无解析警告，全部服务特征与编码器相同。没有 CER/TEDS/OmniDocBench 评分。

历史漂移尚未解决：r6/r7 仍出现异常特征，r7 同二进制 ABBA 候选只有 4/6 正常。最终正常不证明已定位或消除根因，开关继续默认关闭。默认服务没有切换。

最终 r8 医学表格单次 HTTP 29.704 s，其中 decode 26.824 s、25.500 token/s；完整 SDK/PDF 15/30 秒目标未验收。历史 EP FP16 1.362 s 不是本轮严格匹配基线，不能据此宣布追平 EP。

最终 backend SHA256：`5f31f9d86a288d11c3aabb3b567163df70edc919cc3538e32d66a832ac6f898c`。

板端隔离目录：`/home/bianbu/MiniCPM-OCR-1.0/experiments/pc8-attention-20261106-r8/`。

宿主机完整报告、真实渲染图、六文件 overlay、二进制及逐节点 CSV：`D:/MiniCPM-OCR-1.0/results/optimization/20261106-pc8-attention/REPORT-NEXT.md`。本轮继承 `69947e82ae31e4a63df848b6f43949ceab12ffab` 和此前脏工作树；overlay 不是干净 HEAD 的独立完整补丁。未提交、推送或更新飞书。
