# 教师自一致性：agreement 的理论天花板（ml-ranking）

**Created:** 2026-08-19
**Branch:** `feat/ml-ranking`（脚本 `scripts/ml_teacher_self_consistency_probe.py`）
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
- `--require-snapshot` 先从已验证快照行里再做 round-robin 抽样，不会先抽 90
  条全池再丢掉旧行。
- n=90 是成本样本，不是全量池。

## 结果（2026-08-20 snapshot replay）

Wave 1 合进 live writer 之后，用 `--require-snapshot` 从 721 条快照行抽 90 条。
13 个 `(profile_digest, negative_digest)` 组分批回放，快照命中 **90/0**。
钉死 `openai-4`（无回落）。Token Harbor 先前 403 `region_blocked` 是间歇的，
本跑通了。

```text
python scripts/ml_teacher_self_consistency_probe.py \
    --config E:/otherproject/OpenBiliClaw/config.toml \
    --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \
    --instance openai-4 --limit 90 --seed 19 --require-snapshot \
    --out data/ml_artifacts/teacher_self_consistency_v3.json
```

16 次 `discovery.evaluate_batch`（digest 组拆批），约 ¥5.38。90/90 戳记
`openai/deepseek-v4-flash`。不回写分数。

| 量 | 8-19 live 画像 | 8-20 扩池画像 | **8-20 快照回放** |
| --- | --- | --- | --- |
| exact / 快照池 | 1435 / 0 | 5459 / 0 | 6180 / **721** |
| 快照命中 | 0/90 | 0/90 | **90/0** |
| agreement | 0.711（64/90） | 0.667（60/90） | **0.756（68/90）** |
| tp/tn/fp/fn | 20/44/9/17 | 30/30/20/10 | **43/25/11/11** |
| FPR / FNR | 0.170 / 0.459 | 0.400 / 0.250 | 0.306 / 0.204 |
| Spearman ρ / MAE | 0.678 / 0.135 | 0.590 / 0.165 | **0.703 / 0.112** |
| 近门槛带 | 0.286（4/14） | 0.412（7/17） | **0.706（12/17）** |
| 分平台 | yt 0.818 / tw 0.696 / xhs 0.682 / bili 0.652 | yt 0.818 / tw 0.826 / xhs 0.500 / bili 0.522 | yt 0.682 / tw 0.783 / xhs 0.818 / bili 0.739 |
| ML@0.70 / @0.50 | 0.699 / 0.744 | 同左 | 同左 |
| S1.5 | 0.90 | 0.90 | 0.90 |

原 y=1 与复测 y=1 都是 54 条，假阳/假阴对称（11/11）。这是目前最接近「纯采样
噪声天花板」的数：比两次 live-profile 复测高，仍低于 0.90，只略高于 ML@0.50
的 0.744。

## 结果（2026-08-20 live，扩池重测）

同一协议、同一 pinned 实例、同一 seed，池从 exact 1435 扩到 **5459**
（8-20 当天新打 3092 条 exact）。线上 daemon 仍是 `feat/ml-ranking`，
**没有** 写入 `profile_digest` / 快照；`--require-snapshot` 仍会抽空。
90/90 都是当前 effective 画像回放。

```text
uv run --extra dev python scripts/ml_teacher_self_consistency_probe.py \
    --config E:/otherproject/OpenBiliClaw/config.toml \
    --db E:/otherproject/OpenBiliClaw/data/openbiliclaw.db \
    --instance openai-4 --limit 90 --seed 19 \
    --out data/ml_artifacts/teacher_self_consistency_v2.json
```

路由：`default_provider=openai-4`，`fallback_order=['openai-4']`。90/90 复测戳记均为
`openai/deepseek-v4-flash`。7 次 `discovery.evaluate_batch`，约 ¥2.84。
不回写 `discovery_candidates` 分数。

| 量 | 2026-08-19 | 2026-08-20 |
| --- | --- | --- |
| 白名单 / 丢掉 compatible / exact | 2222 / 787 / 1435 | 6246 / 787 / **5459** |
| 快照命中 | 0/90 | 0/90 |
| 教师自洽 agreement | **0.711（64/90）** | **0.667（60/90）** |
| 混淆 tp/tn/fp/fn | 20 / 44 / 9 / 17 | 30 / 30 / 20 / 10 |
| FPR / FNR | 0.170 / 0.459 | **0.400** / 0.250 |
| Spearman ρ / MAE | 0.678 / 0.135 | 0.590 / 0.165 |
| 近门槛带 | 0.286（4/14） | 0.412（7/17） |
| 分平台 | yt 0.818 / tw 0.696 / xhs 0.682 / bili 0.652 | yt 0.818 / tw 0.826 / xhs 0.500 / bili 0.522 |
| ML@0.70 / ML@0.50（对照，未重训） | 0.699 / 0.744 | 同左 |
| S1.5 | 0.90 | 0.90 |

原 y=1 有 40 条、复测 y=1 变成 50 条：当前画像下教师偏松，FPR 被 bili（fp 9）
和 xhs（fp 6）拉高。方向与 8-19 那次「偏严、FNR 高」相反，说明这个数仍然
混着画像漂移，不是稳定的采样噪声上限。

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

**快照回放 agreement 0.756（68/90）**，对照两次 live-profile 的 0.711 / 0.667。
隔离画像漂移之后天花板抬了一截，但仍低于旧 S1.5 的 0.90，只略高于 ML@0.50 的
0.744。继续堆同一套特征挤不出 0.90。**2026-08-21：** 合入门槛改为
0.95 × 本探针快照 agreement（本次 0.756 → 0.718）；见
[`2026-08-21-ml-gate-ranker-separation-spec.md`](./2026-08-21-ml-gate-ranker-separation-spec.md)。

本探针不改 `[discovery].relevance_scorer`，不跳过 `evaluate_batch`。
评估上下文快照的 schema 与回放契约见
[`2026-08-19-ml-eval-context-snapshot-spec.md`](./2026-08-19-ml-eval-context-snapshot-spec.md)。
