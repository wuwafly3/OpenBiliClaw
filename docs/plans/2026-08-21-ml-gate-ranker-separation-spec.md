# Gate / Ranker 分离 Spec — 准入蒸馏不再兼任精排

**Created:** 2026-08-21
**Status:** draft; 修订 [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md)
**Parent:** [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md)
**Also amends:** [`2026-08-15-ml-ranking-plan.md`](./2026-08-15-ml-ranking-plan.md)
**2026-08-24 修订（教师噪声重估，四点）：**
1. **廉价 tags 通道整体移出当前阶段**（父 spec S1.2a 暂缓）：计划不再依赖
   `tag_channel_*` 输入；已落地的通道代码与列保留但视为停用。Phase 1
   接口去掉廉价 tags 项，S1.8 成本模型回到「完整评估仅 y=1 + 不确定带/
   校准集」。
2. **Gate 评估 prompt 无 recent 已核实**（invariant 8）：
   `compact_gate_evaluation_profile_summary` 剔除
   `recent_awareness` / `active_insights` / `speculative_interests` 三键，
   `<profile_recent_context>` 层渲染为空对象，`profile_digest` 对 recent
   改动不变（`tests/test_discovery_engine.py` 钉死）。新合同全量重复对照
   （2026-08-24，918 行，`data/ml_artifacts/gate_contract_self_consistency_*`）：
   两抽一致率 **0.764**（翻转 23.6%），near-threshold 带（±0.05，228 行）
   一致率仅 **0.60** —— 翻转集中在阈值邻域，教师采样噪声主导。
3. **新增跨教师模型对照任务（已定方向：外部模型）**：同一 918 行新合同
   快照换**外部模型**（GPT / Gemini / Claude 等）各跑一遍重复对照，
   分离「同模型采样噪声」与「模型间系统差」；结果决定教师人选与
   S1.5 重订的噪声基数。在结果出来前不换教师、不重训生产 artifact。
4. **S1.5 门槛表整体暂停、待重订**：0.95×自洽公式与 AUC≥0.80 选点被
   2026-08-24 数据证伪（理由见 Phase 1 的暂停说明）。重订前旧表不得作为
   合入/拒绝依据。
**Scope:** 阶段 1 准入 gate 与阶段 2 学习排序的职责切分；gate 训练对画像漂移
的合同；S1.5 / S1.6 验收改写。
**Out of scope:** 打开 live `shadow`/`ml`（除下文锁定的 **explore 去教师**
合同外）、普通策略跳过 y=0 的 `evaluate_batch`、廉价 tags 通道
（2026-08-24 起整体移出当前阶段，见顶部修订 1）、
C2/C3/C6 改用 ranker 分、配置键改名、插件 / desktop / mobile / CLI 推荐 UI。
Explore 去教师是锁定决策，**不**在 Wave A 改运行时。

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
9. **Explore 是第一个可完全去教师的决策点：** 精确 `source_strategy="explore"`
   的准入不再要求教师 `evaluate_batch`。质量底线用 gate 清晰区下沿
   （p > τ_quality，暂定 0.3–0.4）；探索价值 = 投机兴趣挂载命中 **或**
   高多样性。流量池 =
   `(灰区 ∩ 投机命中) ∪ (灰区 ∩ 高多样性)`。灰区暂定
   `[τ_quality, τ_clear]`，τ_clear 暂定 0.85。清晰区高分不走 explore 例外。
   不得把 gate p 写进 `relevance_score`。τ_* 未标定（硬规则 #3），artifact
   缺失 fail-open 回现有教师 0.58。验证：explore 单测在无 LLM 下按公式入池；
   非 explore 路径仍走教师/C1。
10. **评估循环不得在同 tick 内混冻不同 live 画像：**
    `CandidateEvalCoordinator._fill_open_slots` 对本批 worker 只
    `get_profile()` / `capture_live_evaluation_context()` 一次，各 worker
    共用同一 `EvaluationContextSnapshot`。ContextVar 仍隔离不同任务；下一
    fill 才换尺子。验证：并发测试在认知写入期间同一 tick 的
    `profile_digest` 集合大小为 1。
11. **教师负例丢弃页面壳标题：** `recent_negative_exemplars` 不得把
    `document.title` 站点壳（完整网站标题，如 B 站首页
    `哔哩哔哩 (゜-゜)つロ 干杯~-bilibili`）送进教师 prompt。
    名单只列完整标题；被其包含的干杯短标题靠去掉尾部站点品牌匹配，不单列。
    产品名整标题（如 `ChatGLM`）不当壳。事件行可保留；改过滤必须让
    `negative_digest` 跟着变。验证：单测钉死壳标题被丢、内容标题保留。

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

### D5. 并发评估冻到认知写入中的不同版本（已核对）

2026-08-20 同日 13:07–13:11 两次评估的 `recent_awareness` 窗口对打，不是 12h
节拍。原因是打分时画像正在被认知写入换掉。

**已关闭（gate recent）：** `cognition_cycle.py:607-626` 把 `recent_awareness` /
`active_insights` 写回 `soul.json`。Gate compact 已丢掉这三键，这类窗口对打
不再改变 `profile_digest`。

**已关闭（同 tick 尺子）：** `CandidateEvalCoordinator._fill_open_slots` 对本批
最多 3 个 worker 只冻一次 profile + 负例；worker 不再各自 `get_profile()`。
负例不再按 5 分钟缓存在同 fill 内各绑各的。

**仍开着：**

1. 同进程里 soul pipeline 可 `run_if_due` / early-trigger（`pipeline.py:1526`，
   `cognition_cycle.py:379`）。偏好更新、整理、dislike 写回仍会改 compact
   兴趣 / 避雷；**下一 fill** 会看到新 digest（标签仍诚实）。
2. `soul_layer.data.clear(); update; save()`（`cognition_cycle.py:624-626`）
   与那一次 tick 级 `get_profile()` 无读者锁，单次冻结读取仍可能撕到半写入
   画像。未加锁。

标签仍诚实（不同 digest = 不同教师条件）。漏洞是**同一 drain tick 的准入
尺子不一致**，不是 12h 日历。

### D6. 教师负例混入页面壳标题

扩展 dislike 用 `document.title`（`extension/src/content/bilibili.ts:82`）。
首页/壳页面上点踩会把 `哔哩哔哩 (゜-゜)つロ 干杯~-bilibili`、`ChatGLM` 一类
非内容标题送进 `recent_negative_exemplars`（`soul/negative_exemplars.py:56`），
再进教师 `<negative_examples>`。这是采集噪声，不是修辞负例。过滤在 exemplar
装配处做；不删事件。**已落地：** 名单只列完整网站标题；规范化指纹在去掉尾部
ASCII 站点品牌后仍命中被包含的干杯短标题。产品名整标题不进黑名单。

## Priority classification

| Phase | Content | Tier | Why |
| --- | --- | --- | --- |
| 0 | 本文档 + 父 spec/plan 冲突条款作废 | **MUST** | 否则执行者仍按 C1–C7 一次重标定 |
| 1 | 生产 gate 只训验证快照行；GroupKFold by digest | **MUST** | 否则继续蒸馏混合画像的标签 |
| 2 | 画像相对特征（相对 t0 / live compact+负例） | **MUST** | 不变特征则 invariant 4 只是切数据 |
| 3 | S1.5 改相对天花板；Brier 降为测量项 | **MUST** | 0.90 已被 0.756 证伪 |
| 4 | ranker 仍按人类标签 + 曝光账本 | RECOMMENDED | 不阻塞 gate；正样本 < 200 不得进 `shadow` |
| 5 | 廉价 tags 通道 / 跳过 y=0 完整评估 | 父 spec 原计划 | **2026-08-24：廉价 tags 通道移出当前阶段**（顶部修订 1）；全局跳过仍等 S1.5 重订；**explore 去教师**见 Phase 4 |
| 6 | 评估 loop 同 tick 冻一份 snapshot | **MUST**（已落地） | D5：否则 INTEREST 写入仍会撕裂同一轮 |
| 7 | 负例丢弃页面壳标题 | **MUST**（已落地） | D6：否则教师在学站点 chrome |

Wave A = Phase 0–3，可独立交付：文档 + 训练合同 + 特征，**不**改变默认
`relevance_scorer=llm`。Wave B = 父 spec 阶段 2 ranker。Wave C = Phase 4
explore 去教师。D5/D6 已落地。可在 Wave A 之后停止。

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
  recent 层）+ 与 `_get_negative_exemplars` 相同的负例列表。
  **2026-08-24：廉价 tags 移出当前阶段**——live 推理不依赖
  `tag_channel_*`；训练若用 `tags_source=teacher_oracle` 的历史 tags 作
  对照特征，必须在 artifact 里显式标注且不得进入生产 live 特征集。
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

**S1.5 改写（shadow → ml 的硬条件）——2026-08-24 起整体暂停、待重订：**

| 指标 | 门槛 | 说明 |
| --- | --- | --- |
| 准入一致率 | ≥ 0.95 × 同期快照自洽 agreement | 以 `--require-snapshot` 探针为天花板；天花板更新则门槛更新。2026-08-20 天花板 0.756 → 门槛 **0.718**（n=90 是测量，合入前 holdout n≥100） |
| FPR | ≤ 0.10 | 相对教师 y，快照协议 |
| FNR | ≤ 0.15 | 同上 |
| ROC-AUC | ≥ 0.80 | 按 digest 对分组切分 |
| 分平台 | 各自达到一致率与 FPR/FNR | 沿用父 spec；AUC 可作测量 |
| Brier | 测量项，非合入门槛 | 只服务不确定带宽度，不证明能当 curator 分 |

**暂停理由（2026-08-24，依据 918 行新合同重复对照）：**

1. **0.95 × 两抽一致率不是学生上限。** 学生在（多次抽样的）教师标签上学
   到的是每行 P(y=1)；对一次新抽，取 argmax 的最优一致率为
   E[max(p, 1-p)]，严格高于两抽一致率 E[p²+(1-p)²]。按本次翻转结构
   （701 稳定 + 217 翻转），「多数票学生」对单抽的一致率上界 ≈
   (701 + 0.5×217)/918 ≈ **0.882**，而旧公式给出 0.95×0.764 = 0.726。
   旧公式把采样噪声当成了信息上限，既低估上限也缺乏推导。
2. **FPR ≤ 0.10 / FNR ≤ 0.15 对单次噪声抽样不自洽。** 教师两抽互测
   FPR 0.256 / FNR 0.220 —— 教师自己都过不了这两行门槛。评估基准应
   换成多数票标签（≥3 抽）或做噪声校正，否则门槛度量的主要是教师噪声
   而非学生质量。
3. **AUC ≥ 0.80 与部署点脱钩且未做噪声校正。** AUC 是全操作点指标，
   gate 只在 C1 一个阈值上工作，合入依据应是操作点指标；标签翻转
   ~24% 时完美模型对单抽标签的 AUC 上界显著低于 1，0.80 这个数没有
   按噪声上界推导。AUC 应降为测量项。
4. **决策阈选点无标定来源。** 当前 artifact 的 0.5 / 0.7 阈值没有按
   FPR/FNR 成本权衡选点的记录（硬规则 #3）；2026-08-24 snapshot
   artifact @0.7 在其训练样本内一致率仅 ~0.53，说明选点未与任何指标
   闭环。

重订方向（2026-08-24 已锁，数值待跨模型对照后定）：

1. **评估基准 = 多数票教师标签**：同一快照 ≥3 抽（含跨模型抽）取多数票
   作为基准 y。学生学的是平均教师，评估对平均教师；不再对单次抽样
   打合入判定。
2. **合入门槛 = 操作点指标**：仅在校准后的 C1 决策阈一个点上考核
   —— 分平台 FPR / FNR + 置信区间，阈值按「池污染代价 vs 供给损失」
   的成本权衡在校准集上选点并写注释（硬规则 #3）。分平台样本不足
   （如 twitter n≈85）时报告置信区间而非点估计。
3. **AUC / Brier / Spearman / 一致率全部降为测量项**，不再作为合入
   门槛；一致率对照「教师单抽对多数票基准的一致率」作为参照线打印。

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
| 教师模型间系统差 | 跨模型对照（顶部修订 3）的多数票分歧率显著高于同模型两抽 | 换教师 = 重开全部标签合同与 S1.5 噪声基数；先测再换，不得静默混训 |
| 画像条件漂移 | 校准集（live 画像 × 新教师标签）相对天花板的一致率跌破 S1.5 | gate 回落 `shadow`；用新快照行重训 |
| 排序质量漂移 | ranker NDCG 相对手工权重不再显著 | ranker 回落 `weights`；与 gate 开关无关 |

S3.3 原文「校准集 ROC-AUC 跌破则回落 shadow」保留给 **gate**，不得关掉
ranker，也不得把 live-vs-t0 混测当成纯噪声。

### Phase 4 — Explore 去教师（锁定决策；Wave C 才改代码）

**2026-08-21 产品锁：** explore 是全系统第一个可以完全去掉教师 LLM 的决策点。
普通策略仍走教师 C1（0.60），直到全局 skip-y=0 通过 S1.5。

当前代码：`discovery/admission.py:9-10` 仅把精确 `explore` 的教师分门槛降到
0.58；`ExploreStrategy.score_threshold` 同值。预过滤对 explore 放行
（`engine.py:1735`）。教师 prompt 里 explore 是唯一允许的 strategy 例外。

**准入（精确 `source_strategy="explore"`）：**

```
质量底线   := gate_model p > τ_quality          # 清晰区下沿；暂定 0.3–0.4
探索价值   := speculative_hit OR high_diversity
灰区       := τ_quality ≤ p ≤ τ_clear            # τ_clear 暂定 0.85
explore 入池 := 质量底线 AND 探索价值
             ≡ (灰区 ∩ 投机命中) ∪ (灰区 ∩ 高多样性)
```

自洽：gate 灰区是「模型不确定」；投机兴趣是「画像不确定」。Explore 的语义
就是探索不确定，两种不确定性在同一决策点汇合。p > τ_clear 的清晰匹配不占
explore 例外（应交普通 C1 / 非 explore 策略）。p < τ_quality 即使命中投机
或多样性也拒 —— explore 不推垃圾，但也不要求 0.60。

**投机命中（①）：** 复用评估挂载，不新造 embedding 栈。
`engine.py:276-277, 329-337, 3697`：recall pool 现为兴趣权重 49..256，
cosine ≥ `_EVAL_RECALL_MIN_SIMILARITY`（0.45）最多 3 个名字。投机兴趣今天
在 `_active_speculations` / compact recent 层，**不**在 recall pool。Explore
路径把投机 `domain`（及已有 reason 文本）加入**挂载池**（可与 tail 兴趣并列
或 explore 专用侧池），同一 0.45 阈值命中即 `speculative_hit`。不把投机文本
送回 gate 教师 prompt（不变量 8）。

**高多样性（②）：** 方向取反现有疲劳/MMR 余弦。Curator
`recommendation/curator.py:753-769` 已用候选 topic 与 `recent_topic_keys`
的 embedding cosine 做 topic_fatigue；MMR
（`recommendation/engine.py:4873-4948`）是
`α * relevance - β * max_cosine_to_picked`。Explore 多样性 =
`max cos(候选, 近窗已消费) < τ_div`。v1 可用已消费/已看 topic 向量（与
fatigue 同源）；「近 30 天逐条内容 embedding」是标定项，阈值按硬规则 #3
写注释，换 embedding 模型重开。

**教师：** 命中公式的 explore 行跳过 `evaluate_batch`，不写教师
`relevance_score` / `llm_score_raw`。`score_source` 用新的非教师枚举（实现时
定名），不得冒充 `llm`。C2/C3/C6/文案对无教师分 explore 行：**未定**，不得
用 gate p 填 `relevance_score`（不变量 1）。缺 artifact / 特征失败 fail-open
回现有教师 0.58。

**τ_quality / τ_clear / τ_div：** 暂定，须在有生产 gate 校准分之后按硬规则
#3 标定。在此之前本 Phase 只作为合同，不改默认 `relevance_scorer`。

## Expected impact

| Lever | Measured effect |
| --- | --- |
| 阶段 1 不再重标定 C2–C6 | 打开 gate 不改 delight / 通知 / 池内 tier 的分数语义 |
| 快照子集训练 | 去掉空 digest 行的跨期标签噪声（对照：live 复测 0.667 vs 快照 0.756） |
| 相对 S1.5 | 停止用教师自己达不到的 0.90 阻塞成本项 |
| 阶段 2 仍等曝光 | 人类正样本约 50 条，ranker 不提前 |
| Explore 去教师 | 探索流量不再付教师 0.58 评估；灰区 ∩（投机∪多样）才入池 |

## Documentation obligations

- 本文 + 父 spec/plan 冲突条款
- `docs/modules/ml.md` — 职责名改为 gate / ranker；当前 artifact 仍是
  观察性 admission；Explore 去教师为锁定决策，Wave C 才改代码；负例壳标题
  与同 tick 冻 snapshot 已落地
- `docs/changelog.md` 未发布块一条
- 配置键未改：不必改 `docs/modules/config.md` 字段名；补一句「scorer ≠ ranker」
- 不改架构图 / README highlights（尚未上线推理 / 尚未去教师）

## 与父 spec 条款对照（执行者以本表为准）

| 父条款 | 本修订 |
| --- | --- |
| S1.5 Brier ≤ 0.18 合入 | 作废为测量项 |
| S1.6 C1–C7 全部重标定 | 阶段 1 只标定 C1；C7 仍可在 gate 上线后退役 |
| S1.6 C2/C3/C6 同分位映射到 ML 概率 | 作废；继续用教师 `relevance_score` |
| S1.6 C4/C5 top-25 Jaccard | 推迟到阶段 2 |
| S3.3 单一漂移告警 | 拆成教师噪声 / 画像条件 / 排序质量 |
| Wave 0「300 负样本才能训 Wave 1」 | 改为 Wave 2 前置 |
| 全局跳过 y=0 `evaluate_batch` | 仍等 S1.5；**explore** 可按 Phase 4 先行去教师 |
| G1–G4、S0.3a、S1.1、S1.2 泄漏禁令、S1.7 fail-open | 保持 |
