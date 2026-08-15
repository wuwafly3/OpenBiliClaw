# ML Ranking Spec — LLM 蒸馏与学习排序

**Created:** 2026-08-15
**Status:** draft; 阶段 0 为其余阶段的硬前置
**Branch:** `feat/ml-ranking`
**Scope:** `discovery/engine.py` 评估打分、`recommendation/curator.py` 总分、
`recommendation/engine.py` 排序键、admission / delight 阈值标定、曝光与特征日志、
离线训练与回放脚本。

## 1. 目标与非目标

### 目标

- **G1（成本）** 用本地 ML 模型替代绝大多数 `discovery.evaluate_batch` LLM 调用，
  在 relevance 打分一致性门槛内把该 caller 的 token 支出降到当前的 ≤30%。
- **G2（可学习总分）** 把 `PoolCurator` 手工权重的线性总分换成可训练排序模型，
  权重来自数据而非注释里的调参史。
- **G3（LLM 作教师）** LLM 从"每个候选都打分"退到"只标注 ML 不确定的候选 +
  周期性校准集"，形成持续蒸馏回路。
- **G4（硬规则保留）** 所有硬性过滤规则不进模型，保持确定性、可解释、可测试。

### 非目标

- 不做在线学习 / 实时梯度更新。模型离线训练，运行时只做推理。
- 不引入服务端训练或跨用户数据聚合。项目是 local-first 单用户模型。
- 不在阶段 1 声称"推荐质量提升"。阶段 1 只声称成本下降 + 排序等价。

## 2. 现状测量（本节是验收基线，不是设计）

### 2.1 relevance_score 的四重身份

`relevance_score`（0-1）今天同时承担四个互不相同的职责：

| 职责 | 位置 | 依赖形式 |
| --- | --- | --- |
| 入池硬门槛 | `discovery/admission.py:7` `DEFAULT_ADMISSION_MIN_SCORE=0.60`（explore `0.58`） | 绝对阈值 |
| delight 资格 | `recommendation/delight.py` `DEFAULT_DELIGHT_THRESHOLD=0.75`（保守 `0.80`） | 绝对阈值 |
| 总分基项 | `recommendation/curator.py:376` `base = relevance_score * 0.30` | 线性权重 |
| 排序键 / MMR relevance 项 | `recommendation/engine.py:4421`、`engine.py:4895` | 相对序 |

因此**换打分器等于同时动四处语义**。CLAUDE.md 硬规则 #3（阈值标定溯源）
要求 provider/model 更换后重开标定；ML 替换是更彻底的 scorer 更换，
0.60 / 0.58 / 0.75 / 0.80 四个常数必须重新标定，不能沿用。

### 2.2 教师标签可用量

| 来源 | 行数 | 分数区间 | 说明 |
| --- | --- | --- | --- |
| `content_cache.relevance_score` | 1810（全部非零） | **0.60–1.00** | 被 admission 截断，只有入池者 |
| `discovery_candidates`（已评估） | 599 | 0.00–0.90 | 保留 `rejected_low_score` 335 行，**未截断** |
| `evaluator_prefilter_shadow_audit.llm_score` | 967 | 0.00–0.90 | 未截断，且自带 `similarity` 特征 |

`content_cache` 的分布证实了截断（0.6 以下为 0 行）。可用于蒸馏的未截断样本
约 **1566 行**（599 + 967，去重后更少）。两张表都有 30 天保留期
（`prefilter_audit.py:26`、`database.py:405`），阶段 0 必须先固化快照。

`evaluator_prefilter_shadow_audit` 里 `similarity`（profile↔候选 embedding 余弦）
与 `llm_score` 的 Pearson r = **0.2621**（n=967）。单一 embedding 相似度远不足以
复现教师分，但它是已在线计算、零额外成本的强特征。

### 2.3 人类标签可用量与判别力

```
content_cache.feedback_type:  dislike 34 | like 16 | comment 3 | <none> 1757
recommendations.feedback_type: dislike 23 | like 12 | comment  3
events: favorite 1180(positive) | like 63 | feedback negative 28
watch_seconds 有值 65 条（median 29.5s，>30s 32 条）
page_dwell_seconds 有值 65 条（median 36.2s）
view 事件带 (currentTime,duration) 仅 76/1125 条
```

**relevance_score 对 like/dislike 的 AUC = 0.4513，Welch t = -0.046。**
教师信号与用户真实反馈无可测相关性。这不否定蒸馏（蒸馏目标是复现教师以省钱），
但它禁止把阶段 1 表述为质量提升，并且说明 G2 必须用人类标签重训、不能用教师分重训。

### 2.4 曝光漏斗断裂（阶段 0 的根本原因）

```
recommendations: n=1207  presented=0  presented_at=0  feedback_at=38
```

`mark_recommendation s_presented` 只有 `cli.py:12505` 一个调用点；
`api/app.py` 出流路径只读 `presented` 字段（`app.py:6562`、`app.py:7550`），从不写。
浏览器插件 / desktop web / mobile web 三个真实使用面全部不写曝光。

后果：无法构造 (曝光, 未点击) 负样本，任何 learning-to-rank 只能在
"被反馈的 50 条"上训练，且带极强选择偏差。**没有曝光日志，G2 无法验收。**

### 2.5 成本基线（`llm_usage`，834 次调用）

| caller | 调用数 | tokens | CNY | cache hit |
| --- | --- | --- | --- | --- |
| `discovery.evaluate_batch` | 274 | 7,471,592 | 9.1984 | 33% |
| `recommendation.write_expression` | 175 | 4,193,324 | 5.6105 | 38% |
| `soul.preference.chunk` | 227 | 2,091,398 | 3.9015 | 6% |

`discovery.evaluate_batch` 是最大单项，占总 tokens 约 35%。这是 G1 的全部收益空间。

### 2.6 运行时环境

- `numpy 2.1.3` / `scikit-learn 1.6.1` 已安装，但**两者都不在 `pyproject.toml`
  依赖里**（属环境偶然可用）。`lightgbm` 未安装。
- embedding：`qwen3-vl-embedding#dim=1024`，L2 缓存 11,857 行（382MB db + 192MB wal）。
- `profile_digest` 在 2 小时内出现 12 个不同值 —— 画像高频漂移，
  特征必须以"相对当前画像"的形式表达（余弦、rank），不能用绝对 topic 身份。

## 3. 硬性过滤规则边界（保留、不进模型）

以下规则保持确定性代码路径，模型只产出分数，不产出准入决策：

1. `discovery/admission.py::effective_admission_threshold` —— 入池下限（阈值重标定，逻辑不变）。
2. `discovery/engine.py` 时效准入生命周期（`temporal_*`、`reject_temporally_stale_*`）。
3. `recommendation/exclusion.py`、`_exclude_recently_viewed`、
   `_exclude_disliked_topic_candidates*` —— 已看 / 已厌恶排除。
4. `_exceeds_broad_cap` / `_exceeds_amplification_cap` / `_topic_cap` / `_style_cap` /
   `_platform_token` 平台配额 —— 多样性与配额上限。
5. `discovery_candidates` 容量裁剪（`trimmed_capacity`）与 franchise 配额。
6. `_normalize_bonus_per_platform` 跨平台公平归一（见 memory `bonus-cross-platform-fairness`）。

模型分数进入的位置**仅限**：`relevance_score`（阶段 1）与 curator 总分（阶段 2）。

## 4. 分阶段契约

### 阶段 0 — 仪器化（阻塞其余阶段）

**S0.1 曝光日志**：API / 插件 / desktop web / mobile web 四面出流都必须写
`presented=1, presented_at`。这是 CLAUDE.md 陷阱规则 #5（四面契约）的直接适用。

**S0.2 排序特征快照**：新增 `ranking_feature_log` 表，在 serve 时为每个进入
top-K 与被淘汰的候选各记录一行：请求内 rank、模型分、教师分（若有）、
`similarity`、`candidate_tier`、平台、`temporal_class`、`style_key`、
各 bonus 分量、以及是否入选。禁止写标题 / URL / 作者原文（沿用
`evaluator_prefilter_shadow_audit` 的隐私安全先例）。

**S0.3 教师标签快照固化**：把 `discovery_candidates` + `shadow_audit` 的
未截断 (特征, llm_score) 对导出为版本化数据集文件，脱离 30 天保留期。

**S0.4 隐式标签定义**：把 `watch_seconds` / `page_dwell_seconds` / `favorite` /
`like` / `dislike` 归一成单一 `engagement_label`，定义写进本 spec 附录并冻结。

**验收**：连续 7 天真实使用后，曝光行数 > 0 且 (曝光, 无互动) 负样本 ≥ 300 条。

### 阶段 1 — relevance 蒸馏（成本项）

**S1.1** 特征集只用评估时刻已有、零额外网络成本的量：
profile↔候选文本余弦、profile↔封面余弦（多模态开启时）、
互动计数的对数与 `engagement_available` 掩码、时长、发布年龄、
`source_platform` / `source_strategy` / `content_type` / `style_key` one-hot、
`rating_score` / `rating_count` / `source_rank`。**不得**引入需要新 LLM 调用的特征。

**S1.2** 模型形态：离线训练（`[ml]` 可选依赖），运行时**纯 numpy 推理**，
权重以版本化 artifact 落盘。运行时不新增 sklearn / lightgbm 依赖 ——
local-first 桌面分发不接受为推理引入训练框架。

**S1.3** 门控：`[discovery].relevance_scorer = "llm" | "shadow" | "ml"`，
默认 `llm`。`shadow` 下 ML 与 LLM 同时打分、只记录差异，不影响准入。

**S1.4** 一致性门槛（`shadow` 转 `ml` 的硬条件，全部满足才可切）：
- 教师分 vs ML 分 Spearman ρ ≥ 0.75（留出集）；
- top-25 Jaccard ≥ 0.80（同一候选池、同一画像）；
- 在 0.60 准入线上的分类一致率 ≥ 0.90，且假准入率 ≤ 0.10；
- 分平台分层（bilibili / xiaohongshu / bangumi）各自满足上述三条 —— 
  全局达标而 xhs 崩掉不算通过。

**S1.5** 阈值重标定：0.60 / 0.58 / 0.75 / 0.80 四个常数在 ML 分尺度上重新标定，
标定过程与结论写进代码注释（硬规则 #3），并给出与旧分位数对齐的映射证据。

**S1.6** 失败回退：ML 推理异常、artifact 缺失 / 版本不匹配、特征缺失超阈 —— 
一律 fail-open 回落 LLM 路径并 WARNING，绝不静默给默认分
（硬规则 #7：可诊断优于"看起来能跑"）。

**验收**：`discovery.evaluate_batch` token 降到基线的 ≤30%；S1.4 四项门槛全过；
`openbiliclaw cost --by caller` 可见变化；LLM 保留调用集有明确定义。

### 阶段 2 — 学习排序总分（质量项，依赖阶段 0 数据）

**S2.1** 训练目标是阶段 0 的 `engagement_label`，**不是**教师分（见 §2.3）。

**S2.2** 模型替换 `PoolCurator.score_candidates` 的线性组合；
`ScoringWeights` 五项（relevance / freshness / topic_fatigue /
source_monotony / serendipity）从手工常数变为学习到的贡献，
但 fatigue / monotony 的**输入统计量**保持现有确定性计算。

**S2.3** 门控 `[recommendation].ranker = "weights" | "shadow" | "ml"`，默认 `weights`。

**S2.4** 验收必须用人类标签，不得用教师分自证：
留出集上 ML 排序 vs 手工权重排序的 label-weighted NDCG@10 显著更优
（bootstrap 置信区间不含 0），且 explore / 跨平台占比不塌缩
（沿用 `TemporalTopKShadowMetrics` 的分层对照口径）。

**S2.5** 样本量守门：`engagement_label` 正样本 < 200 条时，
阶段 2 不进入 `shadow` 以上状态。当前正样本约 50 条，远未达标。

### 阶段 3 — LLM 教师回路

**S3.1** LLM 只在两种情形调用：ML 分落在准入线邻域的不确定带；
以及固定周期的随机校准集（用于检测 ML 与教师漂移）。

**S3.2** 教师标签持续写入训练集，模型按版本重训并留存评估记录。

**S3.3** 漂移告警：校准集上 Spearman ρ 跌破 S1.4 门槛即自动回落 `shadow`。

## 5. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 蒸馏一个与用户喜好零相关的教师（§2.3） | 阶段 1 只声称成本；质量归阶段 2，用人类标签 |
| 训练数据被 0.60 准入截断 | 只用 `discovery_candidates` + `shadow_audit` 未截断样本 |
| 画像高频漂移（2h 内 12 个 digest） | 特征用相对量（余弦 / rank），禁绝对 topic 身份 |
| 四个阈值语义随 scorer 漂移 | S1.5 强制重标定 + 注释溯源（硬规则 #3） |
| 为推理引入重依赖 | 运行时纯 numpy；训练进 `[ml]` extra |
| 空 / 失败模型结果被缓存 | 沿用硬规则 #2：任何模型产出写库前校验，非法值不落盘 |
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
