# 教师自一致性：agreement 的理论天花板（ml-ranking）

**Created:** 2026-08-19
**Branch:** `feat/ml-ranking-wave1`（脚本 `scripts/ml_teacher_self_consistency_probe.py`）
**问题:** 教师 oracle 准入模型 OOF agreement 只有 0.699（FPR≤0.10 选点）/
0.744（0.50 点），距 S1.5 的 ≥0.90 很远。在改特征或换模型之前，先测
**同一教师再打一遍准入标签能复现多少**——这是 ML-vs-teacher agreement
的理论上限：学生无法比噪声教师更自洽。

## 协议

- 教师白名单：`score_source ∈ {llm, cap_franchise, cap_style}` 且
  `llm_score_raw IS NOT NULL`（与 S1.1 / `resolve_teacher_label` 一致）。
- **身份过滤（硬）**：只保留
  `teacher_model` **完全等于**  pinned 实例的 `provider_type/model`。
  默认实例 `openai-4` → `openai/deepseek-v4-flash`。
  - `openai_compatible/deepseek-v4-flash`（qwen 适配器）排除。
  - `openai/deepseek-v4-flash-thinking`（openai-2）排除。
  - 不按 instance_id 回溯历史行：`teacher_model` 只戳 adapter 名，
    openai / openai-3 / openai-4 在模型名相同时无法区分。
- 复测调用 **只走 `openai-4`**：`ModuleOverride(custom_chain=True, chain=("openai-4",))`
  + `build_llm_registry(..., fallback_order=["openai-4"])`。
  配额/429 **不**静默落到 openai-2 / openai-3。
  若某 chunk 戳出的身份不是 `openai/deepseek-v4-flash`，整 chunk 丢弃，不混入统计。
- 样本：`--limit 90`、`--seed 19`，按 `source_platform` round-robin。
- 评估：优先绑定 `evaluation_context_snapshots`（与行上 `profile_digest` /
  `negative_digest` 配对）。缺快照的历史行才用当前 effective 画像 + 当前
  negative exemplars。`eval_prefilter_mode=off`，跳过 recently-viewed 短路，
  不写回 `discovery_candidates` 分数列，也不把今天的 digest 回填到旧行。
  不同 digest 对不会进入同一 eval batch。生产温度（0.7）保持不变。
- 指标：S1.1 准入线（explore 0.58 / 其余 0.60）上的 agreement / FPR / FNR，
  原始分 MAE 与 Spearman ρ，近门槛带 `|orig-floor|≤0.05`。

## 局限（事前）

- 2026-08-19 那次 live 90 条 **没有** labeling-time 快照：历史行 `profile_digest`
  为空，复测 prompt **不是**原调用的字节级回放，agreement 会 **低估** 纯采样
  噪声天花板。仪表落地后，新教师标签才可按快照回放。
- 原标签可能来自同名模型的 openai / openai-3 / openai-4 任一实例。
- 文本-only 复测（关闭 multimodal）。
- n=90 是成本样本，不是全量 1435。

## 结果（2026-08-19 live）

```text
uv run --extra dev python scripts/ml_teacher_self_consistency_probe.py \
    --config E:/otherproject/OpenBiliClaw/config.toml \
    --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \
    --instance openai-4 --limit 90 --seed 19
```

路由：`default_provider=openai-4`，`fallback_order=['openai-4']`。90/90 复测戳记均为
`openai/deepseek-v4-flash`（无 adapter 混入）。6 次 `discovery.evaluate_batch`
（含成员补修/拆批），约 ¥2.70。不回写 `discovery_candidates` 分数。

| 量 | 值 |
| --- | --- |
| 白名单 / 丢掉 `openai_compatible` / exact 身份 | 2222 / 787 / **1435** |
| 样本（seed 19，四平台 round-robin） | 90（bili 23 / twitter 23 / xhs 22 / yt 22） |
| 教师自洽 agreement | **0.711（64/90）** |
| 混淆 | tp 20 / tn 44 / fp 9 / fn 17 |
| FPR / FNR（原标签为参照） | 0.170 / 0.459 |
| Spearman ρ / MAE | 0.678 / 0.135 |
| 近门槛带 `\|orig-floor\|≤0.05` | 0.286（4/14） |
| 分平台 agreement | yt 0.818 / twitter 0.696 / xhs 0.682 / bili 0.652 |
| ML@0.70 / ML@0.50（对照） | 0.699 / 0.744 |
| S1.5 agreement 门槛 | 0.90 |

原 y=1 有 37 条、复测 y=1 只剩 29 条：当前画像+负例下教师偏严，FNR 被画像漂移放大。

## 如何读这个数

**教师自洽 0.711 与 ML@0.70 的 0.699 同一档，且低于 ML@0.50 的 0.744。**
在当前教师（`deepseek-v4-flash`、生产温度 0.7、画像会漂）下，S1.5 的
agreement≥0.90 **不是模型容量问题，是标签噪声上限**。继续堆同一套特征
也挤不出 0.90。可选下一步：固定实例+降温度重新采教师、或把 S1.5 一致率
改成相对教师自洽的比例，而不是绝对 0.90。

本探针不改 `[discovery].relevance_scorer`，不跳过 `evaluate_batch`。
评估上下文快照的 schema 与回放契约见
[`2026-08-19-ml-eval-context-snapshot-spec.md`](./2026-08-19-ml-eval-context-snapshot-spec.md)。
