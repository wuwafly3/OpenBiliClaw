# ML Ranking 实施计划

**Spec:** `docs/plans/2026-08-15-ml-ranking-spec.md`
**Branch:** `feat/ml-ranking`（worktree `E:\otherproject\OpenBiliClaw-ml-ranking`）
**规则:** 成本门槛不得覆盖一致性门槛；一致性门槛不得覆盖质量门槛。

## Wave 0 — 仪器化（阻塞后续全部 Wave）

1. 修复曝光漏斗：`api/app.py` 出流路径写 `presented=1, presented_at`，
   覆盖插件 / desktop web / mobile web / CLI 四面；补四面回归测试。
2. 新增 `ranking_feature_log` 表 + 写入点（serve 时 top-K 与淘汰各一行），
   隐私安全字段集，30 天保留，沿用 `record_prefilter_shadow_decisions` 的形状。
3. `scripts/export_ranking_dataset.py`：把 `discovery_candidates` +
   `evaluator_prefilter_shadow_audit` 的未截断 (特征, llm_score) 导出为
   版本化数据集，脱离 30 天保留期。
4. 冻结 `engagement_label` 定义（spec 附录 A）并实现纯函数 + 单测。
5. 采集 7 天真实使用数据；负样本 ≥ 300 条方可进 Wave 1 训练。

**门:** 曝光行数 > 0；(曝光, 无互动) 负样本 ≥ 300。

## Wave 1 — relevance 蒸馏

1. `ml/features.py`：确定性纯函数特征提取（spec S1.1 特征集），
   输入 `DiscoveredContent` + 画像视图，输出定长 numpy 向量 + 特征名清单。
2. `pyproject.toml` 新增 `[ml]` extra（训练用 numpy / scikit-learn 显式声明），
   运行时推理只依赖 numpy；numpy 从"环境偶然可用"提升为显式依赖。
3. `scripts/train_relevance_model.py`：离线训练，输出版本化 artifact
   （权重 + 特征名 + 特征版本 + 训练集指纹）。
4. `ml/inference.py`：纯 numpy 推理 + artifact 版本校验 + fail-open 回落 LLM。
5. 配置开关 `[discovery].relevance_scorer = "llm" | "shadow" | "ml"`，默认 `llm`；
   config 校验 + round-trip 测试；API / CLI / RuntimeContext 三个组装根注入。
6. `shadow` 模式：ML 与 LLM 同时打分，差异写 `ranking_feature_log`，不影响准入。
7. `scripts/evaluate_relevance_distillation.py`：算 spec S1.4 四项门槛
   （Spearman / top-25 Jaccard / 准入线一致率 / 分平台分层）。
8. 阈值重标定：0.60 / 0.58 / 0.75 / 0.80 在 ML 尺度上重标，
   注释写标定溯源与分位数对齐证据。
9. 保留 LLM 调用集定义（不确定带 + 校准集）落地为代码常量 + 注释。

**门:** S1.4 四项全过（含分平台分层）；`discovery.evaluate_batch` token ≤ 基线 30%。

## Wave 2 — 学习排序总分

1. `ml/ranker.py`：替换 `PoolCurator.score_candidates` 的线性组合，
   fatigue / monotony 输入统计量保持现有确定性计算。
2. 训练脚本以 `engagement_label` 为目标（**不是**教师分）。
3. 开关 `[recommendation].ranker = "weights" | "shadow" | "ml"`，默认 `weights`。
4. 评估：label-weighted NDCG@10 + bootstrap 置信区间；
   explore / 跨平台占比分层对照（沿用 `TemporalTopKShadowMetrics` 口径）。

**门:** 正样本 ≥ 200 条才可进 `shadow`；NDCG 置信区间不含 0 才可进 `ml`。

## Wave 3 — 教师回路

1. LLM 调用收敛到不确定带 + 周期校准集。
2. 教师标签持续入训练集；模型按版本重训并留存评估记录。
3. 漂移告警：校准集 Spearman ρ 跌破门槛自动回落 `shadow`。

## Wave 4 — 文档与发布

1. `docs/modules/discovery.md` / `recommendation.md` / `config.md` 同步。
2. `docs/changelog.md` 条目。
3. `docs/architecture.md` + `docs/spec.md` §3 + README CN/EN 架构图新增
   ML 推理块与训练 artifact 依赖。
4. README CN/EN 📌 highlights 替换（≤4 条，CN/EN 同步）。

## 当前阻塞项（实测）

- **曝光日志为 0**（`presented=0` / 1207 行）→ Wave 2 无法验收。
- **人类正样本约 50 条**（like 16 + favorite 关联部分）→ 低于 Wave 2 门槛 200。
- **教师分对 like/dislike 的 AUC = 0.4513** → Wave 1 只能声称成本，不能声称质量。
- **未截断教师样本约 1566 行**，且两张来源表都是 30 天滚动保留 → Wave 0 第 3 步必须先做。
