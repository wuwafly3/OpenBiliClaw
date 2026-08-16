# ML Ranking 实施计划

**Spec:** `docs/plans/2026-08-15-ml-ranking-spec.md`
**Branch:** `feat/ml-ranking`（worktree `E:\otherproject\OpenBiliClaw-ml-ranking`）
**规则:** 成本门槛不得覆盖一致性门槛；一致性门槛不得覆盖质量门槛。

## Wave 0 — 仪器化（阻塞后续全部 Wave）

1. ✅ **曝光账本**（commit `de7de536`）：`recommendation_impressions` 表 +
   四面写入（插件 / desktop web / mobile web / CLI），与 `presented` 严格分离。
2. `ranking_feature_log` 表 + 写入点（serve 时 top-K 与淘汰各一行），
   隐私安全字段集，30 天保留，沿用 `record_prefilter_shadow_decisions` 的形状。
3. `scripts/export_ranking_dataset.py`：把 `discovery_candidates`
   未截断 (特征, 分数) 导出为版本化数据集，
   二分类标签按 S1.1 规则在导出时计算并冻结，脱离 30 天保留期。
   **分数溯源已落地（S0.3a）**：`discovery_candidates` 新增 `score_source` /
   `llm_score_raw` 两列（cap 置零前的教师原始分被保留）；导出必须走
   `Database.get_teacher_labeled_discovery_candidates()` 的教师判定白名单，
   标签取 `llm_score_raw`，不得直接 `SELECT relevance_score`（cap 置零行的
   `relevance_score = 0.0` 是配额产物，直接用会产出假负样本）。
   导出时记录批身份（`evaluated_at` 聚簇即批，分组分析与不确定带研究依赖它），
   并经 embedding 服务补算候选文本向量降维投影（≤32 维 PCA/随机投影）入
   数据集——embedding LRU 缓存对历史候选命中率仅 12%，不可作数据源
   （pairwise 原型实测）。
4. 冻结 `engagement_label` 定义（spec 附录 A）并实现纯函数 + 单测。
5. 采集真实使用数据；负样本 ≥ 300 条方可进 Wave 1 训练。

**门:** 曝光行数 > 0；（曝光, 无互动）负样本 ≥ 300。

## Wave 1 — 准入二分类蒸馏（成本项）

1. `ml/features.py`：确定性纯函数特征提取（spec S1.2 特征集，
   含候选文本向量降维投影 ≤32 维——pairwise 原型实测当前特征集的教师
   排序复现上限 ρ≈0.46 / 准入 AUC≈0.72–0.80，文本向量本体是下一特征杠杆），
   输入 `DiscoveredContent` + 画像视图，输出定长 numpy 向量 + 特征名清单。
   **禁** `topic_group` / `style_key` / `franchise_key` / `temporal_*`（标签泄漏；
   多任务原型证实自预测 tags 只能挽回该信息约 4% 增量，两阶段辅助头方案
   已按 2026-08-16 复核降级为证据，见 `docs/plans/2026-08-16-ml-multitask-probe.md`）。
2. `pyproject.toml` 新增 `[ml]` extra（训练用 numpy / scikit-learn 显式声明），
   运行时推理只依赖 numpy；numpy 从"环境偶然可用"提升为显式依赖。
3. `scripts/train_relevance_model.py`：离线训练浅模型（logistic / 浅 GBDT），
   输出版本化 artifact（权重 + 特征名 + 特征版本 + 训练集指纹）。
4. `ml/inference.py`：纯 numpy 推理 + artifact 版本校验 + fail-open 回落 LLM。
5. 配置开关 `[discovery].relevance_scorer = "llm" | "shadow" | "ml"`，默认 `llm`；
   config 校验 + round-trip 测试；API / CLI / RuntimeContext 三个组装根注入。
6. `shadow` 模式：ML 与 LLM 同时判定，分歧写 `ranking_feature_log`，不影响准入。
7. `scripts/evaluate_relevance_distillation.py`：算 spec S1.5 六项门槛
   （AUC / 一致率 / FPR / FNR / Brier / 分平台分层）。
8. 阈值重标定 **C1–C7 全部 7 个常数**（§3.1）：准入线按 FPR/FNR 权衡选点，
   delight/通知/tier 取同分位，注释写标定溯源与分位数对齐证据。
9. LLM 保留调用集定义（y=1 候选 + 不确定带 + 校准集）落地为代码常量 + 注释。
   概率校准（Brier 门槛）用 isotonic——pairwise 原型已验证该形式可用
   （`docs/plans/2026-08-16-ml-pairwise-probe.md`）。

**门:** S1.5 六项全过（含分平台分层）；`discovery.evaluate_batch` token ≤ 基线 70%。

## Wave 2 — 学习排序总分

1. `ml/ranker.py`：替换 `PoolCurator.score_candidates` 的线性组合，
   fatigue / monotony 输入统计量保持现有确定性计算。
2. 训练脚本以 `engagement_label` 为目标（**不是**教师分）。
3. 开关 `[recommendation].ranker = "weights" | "shadow" | "ml"`，默认 `weights`。
4. 评估：label-weighted NDCG@10 + bootstrap 置信区间；
   explore / 跨平台占比分层对照（沿用 `TemporalTopKShadowMetrics` 口径）。

**门:** 正样本 ≥ 200 条才可进 `shadow`；NDCG 置信区间不含 0 才可进 `ml`。

## Wave 3 — 教师回路

1. LLM 调用收敛到三种情形（y=1 候选 + 不确定带 + 周期校准集）。
2. 教师标签持续入训练集；模型按版本重训并留存评估记录。
3. 漂移告警：校准集 ROC-AUC 或准入线一致率跌破门槛自动回落 `shadow`。

## Wave 4 — 文档与发布

1. `docs/modules/discovery.md` / `recommendation.md` / `config.md` 同步。
2. `docs/changelog.md` 条目。
3. `docs/architecture.md` + `docs/spec.md` §3 + README CN/EN 架构图新增
   ML 推理块与训练 artifact 依赖。
4. README CN/EN 📌 highlights 替换（≤4 条，CN/EN 同步）。

## 当前状态（实测）

- ✅ **曝光账本已落地并合并到 main**（commit `de7de536`）。
- **人类正样本约 50 条**（like 16 + favorite 关联部分）→ 低于 Wave 2 门槛 200，
  采集进行中。
- **教师分对 like/dislike 的 AUC = 0.4513** → Wave 1 只声称成本，不声称质量。
- **Wave 1 训练集 ≈ 686 行**（教师判定白名单：非零教师分 599 + shadow_audit
  恢复的 cap 置零真阳性 87；S0.3a 溯源已落地），未截断；来源表 30 天滚动保留
  → Wave 0 第 3 步必须先导出固化。
