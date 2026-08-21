# ML Ranking Spec — LLM 蒸馏与学习排序

**Created:** 2026-08-15
**Status:** draft; 阶段 0 为其余阶段的硬前置。
**2026-08-21 修订：** 阶段 1/2 的职责切分、S1.5 / S1.6 验收与画像漂移训练合同
以 [`2026-08-21-ml-gate-ranker-separation-spec.md`](./2026-08-21-ml-gate-ranker-separation-spec.md)
为准。下文 S1.5 Brier 合入门槛、S1.6「C1–C7 全部重标定」、以及「gate 概率填
curator / MMR / delight」作废。
**Branch:** `feat/ml-ranking`
**Scope:** `discovery/engine.py` 评估打分、`recommendation/curator.py` 总分、
`recommendation/engine.py` 排序键、admission / delight 阈值标定、曝光与特征日志、
离线训练与回放脚本。

## 1. 目标与非目标

### 目标

- **G1（成本）** 用本地 ML 模型替代绝大多数 `discovery.evaluate_batch` LLM 调用，
  在准入判定一致性门槛内把该 caller 的 token 支出降到当前的 ≤70%。
- **G2（可学习总分）** 把 `PoolCurator` 手工权重的线性总分换成可训练排序模型，
  权重来自数据而非注释里的调参史。
- **G3（LLM 作教师）** LLM 从"每个候选都打分"退到"只标注 ML 判为过线的候选 +
  周期性校准集"，形成持续蒸馏回路。
- **G4（硬规则保留）** 所有硬性过滤规则不进模型，保持确定性、可解释、可测试。

### 非目标

- 不做在线学习 / 实时梯度更新。模型离线训练，运行时只做推理。
- 不引入服务端训练或跨用户数据聚合。项目是 local-first 单用户模型。
- 不在阶段 1 声称"推荐质量提升"。阶段 1 只声称成本下降 + 准入判定等价。

## 2. 现状测量（本节是验收基线，不是设计）

### 2.1 relevance_score 的四重身份

`relevance_score`（0-1）今天同时承担四个互不相同的职责：

| 职责 | 位置 | 依赖形式 |
| --- | --- | --- |
| 入池硬门槛 | `discovery/admission.py:7` `DEFAULT_ADMISSION_MIN_SCORE=0.60`（explore `0.58`） | 绝对阈值 |
| delight 资格 | `recommendation/delight.py` `DEFAULT_DELIGHT_THRESHOLD=0.75`（保守 `0.80`） | 绝对阈值 |
| 总分基项 | `recommendation/curator.py:376` `base = relevance_score * 0.30` | 线性权重 |
| 排序键 / MMR relevance 项 | `recommendation/engine.py:4421`、`engine.py:4895` | 相对序 |

四者里**三个是阈值判定，只有一个用到连续序**。这决定了阶段 1 的建模形态：
relevance 在管线中的实际角色是准入门，不是精排分（§2.7 给出经验证据）。

**2026-08-21：** 四处语义不再由同一个 ML 输出承担。Gate 只替换「入池硬门槛」
的判定（C1）；delight / 通知 / 池内 tier（C2/C3/C6）继续用已入池条目的教师
`relevance_score`；curator / MMR（C4/C5）留给阶段 2 ranker。见
[分离 spec](./2026-08-21-ml-gate-ranker-separation-spec.md)。

因此**换打分器不再等于同时动四处语义**。CLAUDE.md 硬规则 #3 仍要求每个被
替换的常数单独标定；阶段 1 只重开 C1。

### 2.2 教师标签可用量

| 来源 | 行数 | 分数区间 | 说明 |
| --- | --- | --- | --- |
| `content_cache.relevance_score` | 1810（全部非零） | **0.60–1.00** | 被 admission 截断，只有入池者 |
| `discovery_candidates`（已评估） | 599 | 0.00–0.90 | 保留 `rejected_low_score` 248 行，**未截断** |
| `evaluator_prefilter_shadow_audit.llm_score` | 967 | 0.00–0.90 | 未截断，且自带 `similarity` 特征 |

`content_cache` 的分布证实了截断（0.6 以下为 0 行）。`discovery_candidates`
是阶段 1 二分类训练集的**唯一来源**；S0.3a 溯源白名单口径下为
**686 行**（非零教师分 599 + 从 shadow_audit 恢复的 cap 置零真阳性 87，
两源分数一致性 max = 0.00）。两张表都有 30 天保留期
（`prefilter_audit.py:26`、`database.py:405`），阶段 0 必须先固化快照。

`evaluator_prefilter_shadow_audit` 里 `similarity`（profile↔候选 embedding 余弦）
与 `llm_score` 的 Pearson r = **0.2621**（n=967）。单一 embedding 相似度远不足以
复现教师分，但它是已在线计算、零额外成本的强特征（§2.7 显示它对准入分组有分离度）。

### 2.3 人类标签可用量与判别力

```
content_cache.feedback_type:  dislike 34 | like 16 | comment 3 | <none> 1757
recommendations.feedback_type: dislike 23 | like 12 | comment  3
events: favorite 1180（positive）| like 63 | feedback negative 28
watch_seconds 有值 65 条（median 29.5s，>30s 32 条）
page_dwell_seconds 有值 65 条（median 36.2s）
view 事件带 (currentTime,duration) 仅 76/1125 条
```

**relevance_score 对 like/dislike 的 AUC = 0.4513，Welch t = -0.046。**
教师信号与用户真实反馈无可测相关性。这不否定蒸馏（蒸馏目标是复现教师以省钱），
但它禁止把阶段 1 表述为质量提升，并且说明 G2 必须用人类标签重训、不能用教师分重训。

### 2.4 曝光漏斗（阶段 0 已修复）

合并前实测：`recommendations` n=1207、`presented=0`、`presented_at=0`、
`feedback_at=38`。`mark_recommendations_presented` 只有 `cli.py` 一个调用点，
API 出流路径只读 `presented` 字段从不写，浏览器插件 / desktop web / mobile web
三个真实使用面全部不写曝光。

后果：无法构造 (曝光, 无互动) 负样本，任何 learning-to-rank 只能在被反馈的
约 50 条上训练，且带极强选择偏差。**没有曝光日志，G2 无法验收。**
已由阶段 0 的曝光账本修复（§S0.1，commit `de7de536`）。

### 2.5 成本基线（`llm_usage`，834 次调用）

| caller | 调用数 | tokens | CNY | cache hit |
| --- | --- | --- | --- | --- |
| `discovery.evaluate_batch` | 274 | 7,471,592 | 9.1984 | 33% |
| `recommendation.write_expression` | 175 | 4,193,324 | 5.6105 | 38% |
| `soul.preference.chunk` | 227 | 2,091,398 | 3.9015 | 6% |

`discovery.evaluate_batch` 是最大单项，占总 tokens 约 35%。这是 G1 的收益空间，
但 S1.8 显示并非全部可省（结构化标注仍需 LLM）。

### 2.6 运行时环境

- `numpy 2.1.3` / `scikit-learn 1.6.1` 已安装，但**两者都不在 `pyproject.toml`
  依赖里**（属环境偶然可用）。`lightgbm` 未安装。
- embedding：`qwen3-vl-embedding#dim=1024`，L2 缓存 11,857 行（382MB db + 192MB wal）。
- `profile_digest` 在 2 小时内出现 12 个不同值 —— 画像高频漂移，
  特征必须以"相对当前画像"的形式表达（余弦、rank），不能用绝对 topic 身份，
  且留出集必须按 `profile_digest` 分组切分。

### 2.7 分数分布：为什么建模成二分类

`scripts/ml_llm_score_diag.py`（只读）在生产库上的实测。

**正式池 `content_cache`（被 0.60 准入截断，n=1810，均值 0.7314）：**

```
0.60→278  0.65→342  0.70→349  0.75→367  0.80→269  0.85→179
0.90→ 22  0.95→  2  1.00→  2
```

**未截断的 `discovery_candidates`（n=599，含被拒行）：**

```
0.1→16  0.2→19  0.3→61  0.4→54  0.5→97  0.6→149  0.7→123  0.8→72  0.9→7
cached                   n=330  range=[0.60, 0.90]  avg=0.713
rejected_low_score       n=248  range=[0.05, 0.58]  avg=0.409
rejected_cache_admission n= 13  range=[0.60, 0.85]  avg=0.735
rejected_franchise_quota n=  8  range=[0.60, 0.90]  avg=0.802
```

三条结论决定建模形态：

1. **分数是量化的，不是连续的。** top-5 档位（0.75 / 0.65 / 0.72 / 0.68 / 0.60）
   覆盖 1810 行中的 741 行（41%）。评估器输出的是几十个离散档位。
2. **可服务区间极窄。** 正式池 86% 落在 `[0.60, 0.80]` 这 0.2 宽度内，
   0.90 以上全池仅 26 行。阈值附近有信息，阈值之上没有 —— 0.62 与 0.78
   对用户没有可测差别（§2.3 的 AUC 0.4513）。
3. **判别力集中在准入线两侧。** 被拒行从 0.05 到 0.58 连续铺开，
   与 admitted 组均值差 0.30 以上；`similarity` 特征在两组间也有分离
   （admitted 0.6132 vs rejected 0.5727，n=967；分平台后 xhs 0.6477 vs 0.5851）。

因此阶段 1 建模为**二分类**（是否达到准入线），而不是连续分回归。
为排序而蒸馏连续分等于蒸馏一个已被证明与用户偏好无关（AUC 0.4513）、
且 41% 集中在 5 个档位上的量化值。

## 3. 准入门槛全清单

管线共 **7 层**门。ML 只替换其中一个判定（第 3 层的分数门），其余全部保持
确定性代码路径 —— 模型产出分数，不产出准入决策。

### 3.1 与分数耦合（换打分器必须重标定）

| # | 门 | 当前值 | 位置 |
| --- | --- | --- | --- |
| C1 | 入池分数门 | `0.60` / explore `0.58` / 可配下限 `0.50` | `discovery/admission.py:7-9` |
| C2 | delight 资格 | `0.75` / 保守 `0.80`（另有 `dynamic_delight_threshold`） | `recommendation/delight.py:31-32` |
| C3 | 主动通知 | `confidence >= 0.82` | `database.py::get_notification_candidate` |
| C4 | 总分 relevance 权重 | `× 0.30` | `recommendation/curator.py:376` |
| C5 | MMR relevance 项 | `α = β = 0.5` | `engine.py:4895` |
| C6 | 池内重排 tier 边界 | `raw_score < 0.92` | `discovery/engine.py:4111` |
| C7 | embedding 预过滤线 | 相似度 `< 0.2`（模式默认 `shadow`，当前不拦） | `discovery/engine.py:175, 171` |

C1–C7 曾是 S1.6 的完整清单。**2026-08-21：** 阶段 1 只标定 C1；C2/C3/C6 留在
教师分尺度；C4/C5 随阶段 2；C7 可在 gate 上线后退役。沿用未替换常数不是阶段 1
失败。

### 3.2 结构规则（与分数无关，ML 不动）

| 层 | 门 | 值 / 判定 | 位置 |
| --- | --- | --- | --- |
| 1 入队 | 每源候选行上限 | `max(target*2, target+120, 600)` | `candidate_pool.py:37` |
| 1 入队 | 历史去重 | `get_existing_discovery_candidate_keys` / `..._content_cache_ids` | 入队前查重 |
| 2 评估前 | 批内 franchise / style 配额 | `4` / `8` | `discovery/engine.py:940, 948` |
| 2 评估前 | related_chain 每 UP 上限 | `3` | `discovery/engine.py:961` |
| 2 评估前 | 评估召回池上限 | `256` | `discovery/engine.py:178` |
| 3 入池 | 时效硬过期 / 待复核 | `evaluate_temporal_eligibility`，置信度阈 `0.80` | `discovery/temporal.py:49` |
| 3 入池 | 已看过 | `_recent_viewed_content_keys` 命中 | `discovery/engine.py:4542` |
| 3 入池 | 池级 franchise 配额 | 同 IP 满 `10` | `discovery/engine.py:956` |
| 3 入池 | 重复 / 容量裁剪 | `rejected_duplicate` / `trimmed_capacity` | `candidate_pool.py:28-33` |
| 4 可服务 | **推荐文案非空** | `TRIM(COALESCE(pool_expression,'')) != ''` | `database.py:916, 943, 7412, 7638` |
| 4 可服务 | pool_status | ∈ `fresh / shown / suppressed` | 同上 |
| 5 serve | topic / 软 topic 配额 | `≤5→1 否则 2` / `≤5→2 否则 3` | `_topic_cap` / `_soft_topic_cap` |
| 5 serve | style / 粗类 / 放大配额 | `max(1,min(3,(limit+1)//3))` / `≤5→2` / `floor(limit*0.25)` | `_style_cap` 等 |
| 5 serve | 平台地板 | 每缺失平台补 `≤5` 条 | `_apply_platform_floor` |
| 5 serve | 厌恶 / 已看 / 时效复核 | 三个 `_exclude_*` | `engine.py:5489-5547` |
| 6 HTTP | 单窗口 franchise 上限 | `≤2`，窗口 `[:20]`（从 40 取） | `api/app.py:7636` |
| 6 HTTP | 首屏补货地板 | 少于 `10` 条触发 serve | `api/app.py:456` |
| — | 跨平台 bonus 归一 | 零为固定点分段归一 | `_normalize_bonus_per_platform` |

模型分数进入的位置**仅限**：gate 的内存字段 `ml_admission_*`（阶段 1，不写
`relevance_score`）与 curator 总分（阶段 2 ranker）。`relevance_score` 仍是教师分。

**标签构造的陷阱**：第 3 层是 6 个并列判定，不是一个。
`rejected_franchise_quota`（8 行，均值 0.802）与 `rejected_cache_admission`
（13 行，均值 0.735）分数**高于**准入线，是被结构规则拒的。
因此二分类标签必须直接用 `relevance_score >= threshold` 计算，
**不得**用 `discovery_candidates.status`，否则这 21 行成为假负样本。

## 4. 分阶段契约

### 阶段 0 — 仪器化（阻塞其余阶段）

**S0.1 曝光账本**（✅ 已落地，commit `de7de536`）：新表
`recommendation_impressions` 记录推荐窗口在插件 / desktop web / mobile web / CLI
四面的真实曝光。**不复用 `recommendations.presented`** —— 该列是未读徽标
（`count_unread_recommendations`）与主动通知（`get_notification_candidate`）的开关，
serve 路径写它会同时清零未读并永久静音主动推送；它也无法表达 rank 与重复曝光。
账本按 `(recommendation_id, surface)` 去重，重复曝光累加 `impression_count`
并保留历史最小 `position`；不含分数列（按 `recommendation_id` 关联
`recommendations.confidence` 即得不可变的曝光时刻分数）。详见
`docs/modules/storage.md` 的 Recommendation Impression Ledger 一节。

**S0.2 排序特征快照**：新增 `ranking_feature_log` 表，在 serve 时为每个进入
top-K 与被淘汰的候选各记录一行：请求内 rank、模型分、教师分（若有）、
`similarity`、`candidate_tier`、平台、`temporal_class`、`style_key`、
各 bonus 分量、以及是否入选。禁止写标题 / URL / 作者原文（沿用
`evaluator_prefilter_shadow_audit` 的隐私安全先例）。对二分类（§2.7）而言，
这是阶段 2 学习排序的特征来源，阶段 1 训练**不依赖**它。

**S0.3 教师标签快照固化**（✅ 初版已落地：
`scripts/export_ranking_dataset.py`——legacy 恢复策略 + 全量 shadow_audit
快照 + 冻结二分类标签；深度学习用特征工程版随 Wave 1 特征定稿扩展）：
把 `discovery_candidates` 未截断 (特征, 分数) 对导出为版本化数据集文件，
脱离 30 天保留期。
二分类标签在导出时按 S1.1 规则计算
（`teacher_score >= effective_admission_threshold`，`teacher_score` 取
S0.3a 溯源白名单的 `llm_score_raw`），导出即冻结，不做二次打标。

**S0.3a 分数溯源（已落地）**：`discovery_candidates.relevance_score` 并非全部来自
教师——intra-batch franchise/style cap 会把 ≥0.5 的分数原地置零、单条评估异常回退
0.0、prefilter enforce 写 `max_sim*0.5` 伪分、recently-viewed / 批截断 / 响应缺成员
也产生非教师 0 分，且 reason-diet 会把多数低分行的 reason 清空，事后无法区分。
现已为每条评估写入 `score_source`（taxonomy 见
`discovery/score_source.py`：`llm` / `cap_franchise` / `cap_style` /
`prefilter` / `viewed` / `eval_error` / `response_missing` / `truncated`，
空串 = 溯源机制上线前的历史行）与 `llm_score_raw`（cap 置零前的教师原始分）。
导出数据集**只允许** `score_source ∈ {llm, cap_franchise, cap_style}` 且
`llm_score_raw IS NOT NULL` 的行，统一走
`Database.get_teacher_labeled_discovery_candidates()`，标签取 `llm_score_raw`
（cap 行的 `relevance_score` 已被置零，不可用作标签）。`content_cache` 行天然
全部是过准入门的教师分，不受此问题影响。同段落地的还有 `teacher_model`
列：每行记录实际应答的 LLM 身份（`provider/model`，取自响应对象）——
**固定教师（deepseek-v4-flash）采集期间按此列筛训练集**，非本教师的
历史行不得进入该教师的门槛拟合；provider 回退换模型的行也会被如实
标记而非静默混入。

**S0.4 隐式标签定义**：把 `watch_seconds` / `page_dwell_seconds` / `favorite` /
`like` / `dislike` 归一成单一 `engagement_label`，定义写进本 spec 附录并冻结。

**验收**：连续 7 天真实使用后，曝光行数 > 0 且 (曝光, 无互动) 负样本 ≥ 300 条。

### 阶段 1 — 准入二分类蒸馏（成本项）

阶段 1 的目标不是复现连续分，而是复现**准入决策**：给定
**(候选, 该次 compact 画像+负例)**，判断 LLM 是否会让它过线。理由见 §2.7
与 [分离 spec](./2026-08-21-ml-gate-ranker-separation-spec.md)。Gate 不兼任
精排；生产训练只接受验证快照行。

**S1.1 标签定义（冻结）**

```
y = 1  if  teacher_score >= effective_admission_threshold(source_strategy)
y = 0  otherwise
```

- **`teacher_score` 的取值来源按 S0.3a 溯源白名单**：教师判定行
  （`score_source ∈ {llm, cap_franchise, cap_style}`）取 `llm_score_raw`
  ——cap 置零行的 `relevance_score` 已被改为 0.0，直接用会把教师判 0.5–0.9
  的行打成假 y=0；溯源机制上线前的历史行（`score_source = ''`）来源未知，
  排除而非猜测。
- 阈值按行取 `effective_admission_threshold`（普通 0.60 / explore 0.58），
  **不是**全局常数 —— explore 行用 0.60 打标会误判。
- 标签**只从分数计算，不看 `status`**：`rejected_franchise_quota` 与
  `rejected_cache_admission` 共 21 行分数高于准入线，是结构规则拒的（§3.2 末）。
- 训练集只用未截断来源：`discovery_candidates` 的教师判定白名单行。
  实测口径（2026-08-16）：非零教师分 599 行 + 从 shadow_audit 恢复的
  cap 置零真阳性 87 行（92% 恢复分 ≥0.5）≈ **686 行**；其中
  `cached` 330 / `rejected_low_score`（非零部分）248 / 结构性拒绝 21 /
  恢复行 87。`content_cache` 的 1810 行全部 `y=1`（0.60 截断），
  **只可作正样本补充，不可单独构成训练集**。
- 按 `(profile_digest, negative_digest)` 分组切分留出集；生产 artifact 要求
  快照 `digests_match()`（§2.6；分离 spec invariant 4）。空 digest 历史行不得
  混入生产训练。

**S1.2** 特征集只用评估时刻已有、零额外网络成本的量：
profile↔候选文本余弦（含 max/mean 与可用性掩码）、profile↔封面余弦（多模态开启时）、
**候选文本向量的降维投影（≤32 维 PCA / 随机投影；评估时 embedding 已计算，
零额外网络成本——原型实测：仅用聚合相似度时教师批内排序只能复现
~0.4，文本向量本体是下一特征杠杆，见
`docs/plans/2026-08-16-ml-pairwise-probe.md`）**、
互动计数的对数与 `engagement_available` 掩码、时长、发布年龄、文本长度统计、
`source_platform` / `source_strategy` / `content_type` one-hot、
`rating_score` / `rating_count` / `source_rank`。
**不得**把完整评估的输出（`topic_group` / `style_key` / `franchise_key` /
`temporal_*`）当特征——那是标签泄漏，且 ML 判定时完整评估尚未发生
（多任务原型实验证实辅助头自预测的 tags 只能挽回该信息约 4% 的增量，
见 `docs/plans/2026-08-16-ml-multitask-probe.md`）；
教师 tags 的正确获取方式是 S1.2a 的廉价标签通道——那是显式采购的输入，
不是隐式泄漏。

**S1.2a 廉价标签通道（双管线，已定案）**：教师 tags 是主信号——
tags oracle 消融（`scripts/ml_tags_ablation_probe.py`，
`docs/plans/2026-08-16-ml-tags-ablation-probe.md`）实测 Δρ = +0.111
（0.604→0.715；AUC 0.799→0.864；xhs ρ +0.210 几乎追平 bilibili），
按事前判定规则（Δρ > 0.05）**选 A**：新增 tags-only 廉价 LLM 通道
（无画像块、无批 rubric、仅输出 topic/style/temporal 三字段）对全量
候选打标，其输出作为 ML 准入判定的输入特征；判定为 y=1 的候选再走
完整 LLM 评估（完整评估自产的标注覆盖廉价版）。**通道单价实测是
Wave 1 的前置任务**——S1.8 的成本模型与 ≤70% 验收线在实测后按
"廉价通道 × 全量 + 完整评估 × y=1 占比"重写。通道输出按教师标注
同款 schema 落 `discovery_candidates`，并带独立标注来源标记
（不混入 `score_source`——那是分数溯源，不是标注溯源）。

**S1.3** 模型形态：离线训练（`[ml]` 可选依赖），运行时**纯 numpy 推理**，
权重以版本化 artifact 落盘。686 行样本（S0.3a 溯源白名单口径）只支持带强正则的浅模型
（logistic / 浅 GBDT），不支持深网。运行时不新增 sklearn / lightgbm 依赖 ——
local-first 桌面分发不接受为推理引入训练框架（纯 numpy 前向/反向/Adam 训练器
原型已验证可行，见 `docs/plans/2026-08-16-ml-pairwise-probe.md`）。
概率校准用 isotonic 回归（artifact 携带映射表，同实验验证）。

**S1.4** 门控：`[discovery].relevance_scorer = "llm" | "shadow" | "ml"`，
默认 `llm`。`shadow` 下 ML 与 LLM 同时判定、只记录分歧，不影响准入。
可行性基线（2026-08-16 实测，同特征集）：logistic AUC 0.724±0.066
（profile_digest 分组切分）/ 0.799±0.028（分层切分），Brier 0.206——
当前特征集距 S1.5 门槛有结构性差距，瓶颈在特征表达力而非建模形态
（pairwise 原型：教师批内排序复现上限 ρ≈0.46）。**文本向量投影、廉价标签通道
（S1.2a）、验证快照训练与画像相对特征落地并实测逼近门槛前，不得离开
`shadow`**（分离 spec Phase 1）。

**S1.5 二分类门槛**（`shadow` 转 `ml` 的硬条件，全部满足才可切）：

| 指标 | 门槛 | 说明 |
| --- | --- | --- |
| ROC-AUC | ≥ 0.80 | 留出集，按 `(profile_digest, negative_digest)` 分组切分 |
| 准入线一致率 | ≥ 0.95 × 快照自洽 agreement | **2026-08-21：** 绝对 0.90 作废。2026-08-20 天花板 0.756 → 门槛 0.718；holdout n≥100。见分离 spec S1.5 |
| 假准入率（FPR） | ≤ 0.10 | LLM 会拒、ML 放行 —— 直接污染池 |
| 假拒率（FNR） | ≤ 0.15 | LLM 会收、ML 拦掉 —— 供给损失，比 FPR 可容忍 |
| 校准 | Brier 为测量项 | **2026-08-21：** 不再为 C4/C5/C6 合入。只服务不确定带宽度 |
| 分平台分层 | 各自达到一致率与 FPR/FNR | AUC 可作测量 |

FPR 比 FNR 收得更紧，因为假准入把坏内容推到用户面前，假拒只是少几条供给
（补货机制会补上）。分层是硬要求：全局达标而 xhs 崩掉不算通过。

**S1.6 阈值重标定（2026-08-21 收窄）：** 阶段 1 只重开 **C1**。决策阈按 S1.5
的 FPR/FNR 权衡选择，注释写标定（硬规则 #3）。C2 / C3 / C6 继续用已入池
教师 `relevance_score`，禁止把旧分位映射到 gate 概率。C4 / C5 推迟到阶段 2
ranker，不做 top-25 Jaccard 对齐。C7 若 gate 上线可退役。详见
[分离 spec](./2026-08-21-ml-gate-ranker-separation-spec.md)。

**S1.7** 失败回退：ML 推理异常、artifact 缺失 / 版本不匹配、特征缺失超阈 ——
一律 fail-open 回落 LLM 路径并 WARNING，绝不静默给默认判定
（硬规则 #7：可诊断优于"看起来能跑"）。

**S1.8 LLM 保留调用集（S1.2a 定案后的形态）**：ML 替换分数门，
标签信息由廉价 tags-only 通道供给（S1.2a）。LLM 调用收敛为三类：
① 廉价标签通道（全量候选，tags-only prompt，单价待实测）；
② 完整评估（仅 y=1 候选——入池需要完整分 + 结构化标注 + pool 文案，
其自产标注覆盖①的版本）；
③ 维护性调用（概率落在决策阈邻域的不确定带 + 周期性随机校准集）。
成本收益 = "不再为注定被拒的候选付**完整评估**费用"，但新增①的
全量廉价成本——**≤70% 验收线在①单价实测后按新成本模型重写**，
实测前该数字仅为占位。教师批间漂移（std 0.114 ≈ 批内 0.137，
pairwise 原型）说明准入线邻域的教师判定本身带噪，不确定带宽度
的标定要以此为依据而非拍脑袋。

**验收**：`discovery.evaluate_batch` token 降到基线的 ≤70%（对应上述理论上界的
保守取值，非 ≤30% —— 结构化标注调用无法省去）；S1.5 按分离 spec 改写后的
门槛全过（含分平台 FPR/FNR）；**仅 C1** 重标定且有注释溯源；
`openbiliclaw cost --by caller` 可见变化。

### 阶段 2 — 学习排序总分（质量项，依赖阶段 0 数据）

**S2.1** 训练目标是阶段 0 的 `engagement_label`，**不是**教师分（见 §2.3）。

**S2.2** 模型替换 `PoolCurator.score_candidates` 的线性组合；
`ScoringWeights` 五项（relevance / freshness / topic_fatigue /
source_monotony / serendipity）从手工常数变为学习到的贡献，
但 fatigue / monotony 的**输入统计量**保持现有确定性计算。
**2026-08-21：** 默认不把 `ml_admission_p` 或教师分当 ranker 主特征。

**S2.3** 门控 `[recommendation].ranker = "weights" | "shadow" | "ml"`，默认 `weights`。

**S2.4** 验收必须用人类标签，不得用教师分自证：
留出集上 ML 排序 vs 手工权重排序的 label-weighted NDCG@10 显著更优
（bootstrap 置信区间不含 0），且 explore / 跨平台占比不塌缩
（沿用 `TemporalTopKShadowMetrics` 的分层对照口径）。

**S2.5** 样本量守门：`engagement_label` 正样本 < 200 条时，
阶段 2 不进入 `shadow` 以上状态。当前正样本约 50 条，远未达标。

### 阶段 3 — LLM 教师回路

**S3.1** LLM 只在三种情形调用（见 S1.8）：二分类判定为 y=1 的候选（取结构化
标注）；概率落在决策阈邻域的不确定带；固定周期的随机校准集。

**S3.2** 教师标签持续写入训练集，模型按版本重训并留存评估记录。

**S3.3** 漂移告警（2026-08-21 拆分）：教师采样噪声 / 画像条件漂移 / 排序质量
分开处理。校准集一致率跌破 S1.5 只回落 **gate** 的 `shadow`，不关 ranker。
详见分离 spec Phase 3。

## 5. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 蒸馏一个与用户喜好零相关的教师（§2.3） | 阶段 1 只声称成本；质量归阶段 2，用人类标签 |
| 训练数据被 0.60 准入截断 | 只用 `discovery_candidates` 未截断样本；`content_cache` 仅作正样本补充 |
| 用 `status` 打标引入假负样本 | S1.1 强制用 `relevance_score >= 阈值` 计算标签 |
| explore 行按 0.60 打标 | S1.1 按行取 `effective_admission_threshold`（0.58） |
| LLM 输出字段当特征（标签泄漏） | S1.2 显式禁 `topic_group` / `style_key` / `franchise_key` / `temporal_*` |
| 画像高频漂移（2h 内 12 个 digest） | **2026-08-21：** 生产 gate 只训验证快照行；特征相对 t0/live compact+负例；留出按 digest 对切分。见分离 spec D2 / invariant 4 |
| 七个阈值语义随 scorer 漂移 | 阶段 1 只标定 C1；其余常数不随 gate 换尺度 |
| 误以为能省下全部评估调用 | S1.8：结构化标注仍需 LLM，验收改为 ≤70% 而非 ≤30% |
| 为推理引入重依赖 | 运行时纯 numpy；训练进 `[ml]` extra |
| 空 / 失败模型结果被缓存 | 沿用硬规则 #2：任何模型产出写库前校验，非法值不落盘 |
| 686 行样本过拟合自证 | 浅模型 + 强正则；按画像版本分组留出；分平台分层验收 |
| 样本量不足导致过拟合自证 | S2.5 正样本门槛；留出集 + bootstrap 区间 |

## 6. 文档与四面同步义务

按 CLAUDE.md 文档要求，落地时必须同步：
`docs/modules/discovery.md`、`docs/modules/recommendation.md`、
`docs/modules/config.md`（新增两个 scorer/ranker 开关）、
`docs/changelog.md`、`docs/architecture.md` + `docs/spec.md` §3 + README CN/EN
架构图（新增 ML 推理块与训练 artifact 依赖）。
阶段 0 的曝光写入必须覆盖插件 / desktop web / mobile web / CLI 四面，
或在 PR 里明确写出排除项。

## 附录 A — engagement_label 定义（阶段 0 冻结）

待阶段 0 实现时以实测分布填入并冻结；候选形式：
`favorite`/`like`/`coin` → 强正；`watch_seconds ≥ 0.5 * duration` 或
`page_dwell_seconds ≥ 120` → 弱正；曝光且无任何互动 → 负；
`dislike`/`feedback:negative` → 强负；曝光缺失 → 丢弃（不可作负样本）。
