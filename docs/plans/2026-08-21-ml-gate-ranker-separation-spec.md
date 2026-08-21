# Gate / Ranker 分离 Spec — 准入蒸馏不再兼任精排

**Created:** 2026-08-21
**Status:** draft; 修订 [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md)
**Parent:** [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md)
**Also amends:** [`2026-08-15-ml-ranking-plan.md`](./2026-08-15-ml-ranking-plan.md)
**Scope:** 阶段 1 准入 gate 与阶段 2 学习排序的职责切分；gate 训练对画像漂移
的合同；S1.5 / S1.6 验收改写。
**Out of scope:** 打开 live `shadow`/`ml`、跳过 y=0 的 `evaluate_batch`、廉价
tags 通道落地、C2/C3/C6 改用 ranker 分、配置键改名、插件 / desktop / mobile /
CLI 推荐 UI。

## Goal

2026-08-15 spec 已经把阶段 1 写成二分类准入，却仍要求 gate 概率去填
curator / MMR / delight 的连续槽（S1.5 Brier、S1.6 的 C2–C6）。同时训练把
不同 `profile_digest` 的教师行当 i.i.d.，而教师 prompt 本身是
**(compact 画像, 负例, 候选)** 三元组。结果是：学生在学一个会随画像移动的
老师，又被要求把这个不稳定的概率拿去排序。

本修订的可验证结果：

- **身份分离：** gate 只产出准入决策（及可选的不确定带），**不写**
  `relevance_score`，不进入 `PoolCurator` / MMR。ranker 只在已入池集合上
  学习排序，**不以教师分为监督**。
- **画像条件：** 生产 gate artifact 只在带验证快照的教师行上训练；特征是
  `(候选, 打标时刻 compact 画像+负例)` 的纯函数。推理用**当前**同一序列化。
- **门槛改写：** 准入一致率对照教师快照自洽天花板，而不是绝对 0.90。
  复现：`scripts/ml_teacher_self_consistency_probe.py --require-snapshot`
  （2026-08-20：agreement **0.756**，n=90）。

## Design invariants (MUST hold in every phase)

1. **Gate 不写 `relevance_score`：** `AdmissionModel` 只挂
   `DiscoveredContent.ml_admission_*`（内存）。`relevance_score` /
   `llm_score_raw` 仍只来自教师路径（`score_source ∈ {llm, cap_franchise,
   cap_style}`）。验证：discovery 评估 / ML 推理单测不得把 ML 概率赋给
   `relevance_score`。
2. **Ranker 不以教师分为 y：** `PoolCurator.score_candidates` 的替换模型
   只吃阶段 0 的 `engagement_label`（父 spec 附录 A）。教师 `llm_score_raw`
   不得进入 ranker 损失。验证：训练脚本单测拒绝 `teacher_score` 列当标签。
3. **两个开关独立：** `[discovery].relevance_scorer` 只动评估是否调用 /
   跳过完整 LLM；`[recommendation].ranker` 只动 serve 前总分。打开其一不得
   改变另一条路径的分数语义。配置 round-trip 单测锁这个边界。
4. **Gate 是画像条件模型：** 训练样本是
   `(candidate, snapshot_t0) → y_t0`，其中 `y_t0` 按 S1.1 从该行
   `llm_score_raw` 计算。推理样本是
   `(candidate, live compact profile + live negatives) → P(admit)`。
   缺快照或 digest 为空的历史行不得进入生产 artifact。验证：训练入口
   `--require-snapshot`（比 `--require-profile-digest` 更严：必须能
   `get_evaluation_context_snapshot` 且 `digests_match()`）。
5. **快照字节与评教师 prompt 切片一致：** `profile_digest` /
   `negative_digest` 与 eval cache 同哈希（`discovery/eval_context.py`，
   `tests/test_eval_context.py::test_profile_digest_matches_engine_payload_shape`）。
   改 compact 视图或负例 schema 必须 bump 特征版本，不得 silently 混训。
6. **结构规则仍在模型之外：** 父 spec §3.2 全部保持确定性代码路径。gate
   产出概率，不产出 `rejected_franchise_quota` 这类决策。
7. **Fail-open：** artifact 缺失、特征版本不匹配、廉价 tags 缺失、编码失败
   → 不改 admission，回落完整 LLM，打 WARNING（CLAUDE.md 硬规则 #7）。
8. **Gate 评估画像不含 recent 层：** `_evaluation_profile_summary` /
   `profile_digest` / 新写入的 `evaluation_context_snapshots` 使用
   `compact_gate_evaluation_profile_summary`：在 compact 上限之上丢掉
   `recent_awareness` / `active_insights` / `speculative_interests`。
   推荐表达与未来 ranker 仍走 `compact_content_prompt_profile_summary`
   （含 recent）。验证：`tests/test_discovery_engine.py` 改 recent 不得改
   digest；`tests/test_recommendation_engine.py` 的推荐摘要仍含 recent。
   历史快照若仍带 recent 键，属旧教师合同，不得与新切片混成同一
   `FEATURE_VERSION` 生产 artifact。

## Current diagnosis

### D1. `relevance_score` 四身份把 gate 焊死在 ranker 槽上

父 spec §2.1 已列出四处用途。阶段 1 正确选择了二分类，但 S1.5 把
Brier ≤ 0.18 写成合入门槛，理由是「概率要当分数用于 C4 / C5 / C6」
（`docs/plans/2026-08-15-ml-ranking-spec.md`）。
`recommendation/curator.py` 的 `base = relevance_score * 0.30` 与
discovery 准入线不是同一职责。用同一 ML 输出同时做「会不会过 0.60」和
「入池后排第几」，会把教师噪声和画像漂移送进精排。

**本修订：** C1 归 gate。C4 / C5 归 ranker，阶段 1 不碰。C2 / C3 / C6 继续
用**已入池条目的教师 `relevance_score`**（y=1 仍走完整评估，见父 spec S1.8），
阶段 1 不做分位映射到 gate 概率。

### D2. 教师标签随画像漂移，训练合同未写

教师 user 消息不是「标题 vs 静态兴趣表」。生产路径
`ContentDiscoveryEngine._evaluate_batch_once` 喂给
`build_batch_content_evaluation_prompt` 的是：

| 块 | 内容 | 漂移速度 |
| --- | --- | --- |
| `<profile_core>` … `<profile_style_context>` | compact 人格 / 兴趣 / 风格 | 慢 |
| `<profile_recent_context>` | **空对象**（gate 不再发送 awareness / insights / speculations） | 不变 |
| `<negative_examples>` | 最多 16 条 `{title, reason, age_days}` | 中（14d 半衰期，5min 缓存） |
| `<evaluation_context>` | `evaluated_at` | 每批 |
| `<content_batch>` | sparse-json：title / author / 正文 / 互动 / tags / 可选 `related_interests` / 封面锚点 | 每候选 |
| 不进 prompt | `personality_portrait`、常看 UP | — |

同一 pinned 教师：live 画像复测 agreement 0.711 → 0.667；快照回放 **0.756**
（`docs/plans/2026-08-19-ml-teacher-self-consistency-probe.md`）。
差的是画像+负例，不是采样噪声本身。

快照 writer（`evaluation_context_snapshots`）只解决了**复测能否回放 t0**。
`scripts/train_relevance_model.py` 仍把多 digest 行混成一张表，特征几乎全是
候选侧统计 + 标签 one-hot + 一条 `sim`。S3.3「漂移告警」指校准集 AUC，
不是用户画像变了标签还能否用。

### D3. Wave 0 第 5 步误把人类负样本当作 gate 前置

父 plan Wave 0：「（曝光, 无互动）负样本 ≥ 300 方可进 Wave 1」。
那是 **ranker** 的样本量守门（父 spec S2.5），不是 gate。gate 的标签是教师
`y`，来源 `discovery_candidates` 白名单。两套门槛不得互相阻塞。

### D4. 当前学生看不到教师真正在用的条件

`ml/features.py` `FEATURE_VERSION = admission-teacher-tags-v1`：平台 / 策略 /
长度 / 互动 log1p / cheap-or-oracle tags / `sim`。没有 dislike 命中、负例
话术重叠、style 偏好 vs `style_key`、recall-pool `related_interests`。
在 D2 的输入清单下，agreement 贴近教师自洽是偶然，不是合同。

## Priority classification

| Phase | Content | Tier | Why |
| --- | --- | --- | --- |
| 0 | 本文档 + 父 spec/plan 冲突条款作废 | **MUST** | 否则执行者仍按 C1–C7 一次重标定 |
| 1 | 生产 gate 只训验证快照行；GroupKFold by digest | **MUST** | 否则继续蒸馏混合画像的标签 |
| 2 | 画像相对特征（相对 t0 / live compact+负例） | **MUST** | 不变特征则 invariant 4 只是切数据 |
| 3 | S1.5 改相对天花板；Brier 降为测量项 | **MUST** | 0.90 已被 0.756 证伪 |
| 4 | ranker 仍按人类标签 + 曝光账本 | RECOMMENDED | 不阻塞 gate；正样本 < 200 不得进 `shadow` |
| 5 | 廉价 tags 通道 / 跳过 y=0 完整评估 | 父 spec 原计划 | 本修订不重开；成本线仍待通道单价 |

Wave A = Phase 0–3，可独立交付：文档 + 训练合同 + 特征，**不**改变默认
`relevance_scorer=llm`。Wave B = 父 spec 阶段 2 ranker。可在 Wave A 之后停止。

## Phase designs

### Phase 0 — 合同落地（文档）

- 父 spec **Status** 指向本文。S1.5 的 Brier 行、S1.6「C1–C7 全部重标定」、
  「模型分数进入 relevance + curator 总分」三条作废，改引本节。
- 父 plan Wave 0 第 5 步改为 Wave 2 前置；Wave 1 第 7–8 步去掉 C2–C6 与
  Brier 合入门槛。
- 配置键暂不改名（`relevance_scorer` / `ranker` 已是两条开关）。文档与
  `docs/modules/ml.md` 用 **gate / ranker** 称呼职责，避免再把 gate 叫
  ranker。

**验收：** 读父 spec 阶段 1 的人会落到本文；父 plan Wave 1 不再把 C1–C7
当作合入门。

### Phase 1 — Gate（准入蒸馏，成本项）

**任务句（冻结为 A）：** 预测「教师在 **该次 prompt 切片** 下会不会过
`effective_admission_threshold`」。不是学一个与画像无关的内容先验。

**接口：**

- Consumes: `DiscoveredContent` + 与 `_evaluation_profile_summary` 相同的
  **gate compact** dict（`compact_gate_evaluation_profile_summary`，无
  recent 层）+ 与 `_get_negative_exemplars` 相同的负例列表 + 廉价 tags
  （live 用 `tag_channel_*`，oracle 训练可标明 `tags_source`）。
- Produces: `ml_admission_p` / 决策阈上的 0/1。不写库分。
- 完整评估：仅 gate=1、不确定带、周期校准集（父 spec S1.8 仍成立）。
  过线候选的 `relevance_score` 仍是教师分，供 C2 / C3 / C6 与文案。

**标签：** 父 spec S1.1 不变。额外过滤：

```
score_source ∈ {llm, cap_franchise, cap_style}
llm_score_raw IS NOT NULL
teacher_model 与 pinned 教师一致（沿用自洽探针的精确身份规则）
profile_digest / negative_digest 非空
evaluation_context_snapshots 能取到且 digests_match()
```

**切分：** GroupKFold / holdout 按 `(profile_digest, negative_digest)`，
同一对不得跨 train/holdout。

**S1.5 改写（shadow → ml 的硬条件）：**

| 指标 | 门槛 | 说明 |
| --- | --- | --- |
| 准入一致率 | ≥ 0.95 × 同期快照自洽 agreement | 以 `--require-snapshot` 探针为天花板；天花板更新则门槛更新。2026-08-20 天花板 0.756 → 门槛 **0.718**（n=90 是测量，合入前 holdout n≥100） |
| FPR | ≤ 0.10 | 相对教师 y，快照协议 |
| FNR | ≤ 0.15 | 同上 |
| ROC-AUC | ≥ 0.80 | 按 digest 对分组切分 |
| 分平台 | 各自达到一致率与 FPR/FNR | 沿用父 spec；AUC 可作测量 |
| Brier | 测量项，非合入门槛 | 只服务不确定带宽度，不证明能当 curator 分 |

C1 决策阈仍按 FPR/FNR 权衡选择，注释写标定（硬规则 #3）。**不**把旧
delight 0.75 / 通知 0.82 / tier 0.92 映射到 gate 概率。

**画像相对特征（进 `ml` 前 MUST）：** 至少包含可在 snapshot JSON 上重算的
相对量，例如：profile↔候选文本余弦（对 t0 summary 的可嵌入文本视图）、
`disliked_topics` 命中、负例标题的结构重叠、cheap `style_key` 与画像
`style.*` 的对齐。仍禁止把教师 `topic_group` / `style_key` /
`franchise_key` / `temporal_*` 当 live 特征（父 spec S1.2）。`sim` 若来自
打标时刻 prefilter audit，算相对特征，但不得当唯一画像条件。

**特征版本：** 引入相对特征后 bump `FEATURE_VERSION`，旧
`admission-teacher-tags-v1` 仅作对照，不得再当生产 artifact。

### Phase 2 — Ranker（学习排序，质量项）

父 spec 阶段 2 保持，补充边界：

- 输入：已过 §3.2 结构规则且可服务的池条目；疲劳 / 单调的**统计量**保持
  现有确定性计算（`PoolCurator`）。Serve 时可读 live compact **含
  recent 层**（`compact_content_prompt_profile_summary`）；recent 不得回灌
  gate 教师 prompt 或 gate 特征。
- 监督：`engagement_label` only。曝光缺失不得当负样本（父 spec 附录 A）。
- 不得把 `ml_admission_p` 或教师分当作 ranker 的主特征去「蒸馏教师序」
  ——父 spec §2.3 教师 vs like/dislike AUC 0.4513 仍然有效。gate 概率若进
  ranker，只能当一个普通特征并在消融里证明有增量，默认**不进**。
- 开关与样本量守门不变：正样本 < 200 不得进 `ranker=shadow`。
- C4 / C5 在 ranker 进入 `ml` 时由数据重训，不在 gate 阶段做 Jaccard 对齐。

### Phase 3 — 两类漂移，分开告警

| 种类 | 信号 | 动作 |
| --- | --- | --- |
| 教师采样噪声 | 快照自洽探针 agreement 下跌 | 重开教师温度 / 换 pinned 实例；不立刻改 gate 特征 |
| 画像条件漂移 | 校准集（live 画像 × 新教师标签）相对天花板的一致率跌破 S1.5 | gate 回落 `shadow`；用新快照行重训 |
| 排序质量漂移 | ranker NDCG 相对手工权重不再显著 | ranker 回落 `weights`；与 gate 开关无关 |

S3.3 原文「校准集 ROC-AUC 跌破则回落 shadow」保留给 **gate**，不得关掉
ranker，也不得把 live-vs-t0 混测当成纯噪声。

## Expected impact

| Lever | Measured effect |
| --- | --- |
| 阶段 1 不再重标定 C2–C6 | 打开 gate 不改 delight / 通知 / 池内 tier 的分数语义 |
| 快照子集训练 | 去掉空 digest 行的跨期标签噪声（对照：live 复测 0.667 vs 快照 0.756） |
| 相对 S1.5 | 停止用教师自己达不到的 0.90 阻塞成本项 |
| 阶段 2 仍等曝光 | 人类正样本约 50 条，ranker 不提前 |

## Documentation obligations

- 本文 + 父 spec/plan 冲突条款
- `docs/modules/ml.md` — 职责名改为 gate / ranker；当前 artifact 仍是
  观察性 admission
- `docs/changelog.md` 未发布块一条
- 配置键未改：不必改 `docs/modules/config.md` 字段名；补一句「scorer ≠ ranker」
- 不改架构图 / README highlights（尚未上线推理）

## 与父 spec 条款对照（执行者以本表为准）

| 父条款 | 本修订 |
| --- | --- |
| S1.5 Brier ≤ 0.18 合入 | 作废为测量项 |
| S1.6 C1–C7 全部重标定 | 阶段 1 只标定 C1；C7 仍可在 gate 上线后退役 |
| S1.6 C2/C3/C6 同分位映射到 ML 概率 | 作废；继续用教师 `relevance_score` |
| S1.6 C4/C5 top-25 Jaccard | 推迟到阶段 2 |
| S3.3 单一漂移告警 | 拆成教师噪声 / 画像条件 / 排序质量 |
| Wave 0「300 负样本才能训 Wave 1」 | 改为 Wave 2 前置 |
| G1–G4、S0.3a、S1.1、S1.2 泄漏禁令、S1.7 fail-open | 保持 |
