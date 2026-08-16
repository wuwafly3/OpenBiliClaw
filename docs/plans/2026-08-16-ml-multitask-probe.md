# ML 多任务蒸馏原型实验（ml-ranking 问题 2）

**Created:** 2026-08-16
**Branch:** `feat/ml-multitask-probe`（脚本 `scripts/ml_multitask_distill_probe.py`）
**问题:** spec S1.1 把 `style_key` / `temporal_class` 列为输入特征，但两者是
LLM 评估的**输出**——ML 推理时没有 LLM，这是特征穿越。本实验验证多任务蒸馏
（联合/两阶段预测辅助标签）能否替代这两个特征。

## 数据集（真实库，只读）

- 686 行未截断教师标签：599 行来自 `discovery_candidates.relevance_score`
  （非零分数天然是教师判断——cap 只置零不降分），**87 行是被 cap 置零的样本、
  教师分从 `evaluator_prefilter_shadow_audit.llm_score` 按评估时刻恢复**
  （92% 恢复分 ≥0.5，正是 cap 置零签名；两源一致性 |persisted−audit| max=0.00）。
- 特征 55 维：shadow-audit 相似度聚合（last/max/min/mean/std + 可用性掩码）、
  10 类互动计数的 log1p + 掩码、时长、发布年龄、标题/简介/正文长度、
  平台/策略/context/content_type one-hot。**不含任何教师输出特征。**
- 辅助标签：style 12 类（<10 行并入 other）、temporal 5 类（breaking 并入 current）。
- 协议：StratifiedKFold(5)×3 seeds，分层键 = 平台×style；报告总体与分平台
  Spearman ρ。

## 结果

| 模型 | Spearman ρ | rho_bilibili | rho_xiaohongshu |
| --- | --- | --- | --- |
| Ridge（单任务基线） | 0.612±0.041 | 0.566 | 0.447 |
| HistGradientBoosting | 0.587±0.053 | 0.575 | 0.414 |
| MLP 单任务 | 0.584±0.032 | 0.535 | 0.443 |
| MLP 联合多任务（原始分数目标） | **0.452±0.075** | 0.426 | 0.299 |
| MLP 联合多任务（分数目标 z-score） | 0.611±0.040 | 0.553 | 0.463 |
| **两阶段（辅助分类器概率作特征）** | **0.616±0.042** | 0.582 | 0.450 |

辅助头（held-out 准确率 vs 多数类基线）：style 0.391 vs 0.257（+13pp）、
temporal 0.495 vs 0.421（+7pp）。

消融：去掉全部相似度特征 ρ 0.605→0.580（−0.025）。

## 结论

1. **辅助任务可学**：style/temporal 头显著超过多数类基线，特征空间确实
   携带教师用于打这两个标签的信息——多任务蒸馏的数据基础成立。
2. **当前数据量下无增益**：两阶段 0.616 vs Ridge 0.612，差值远小于 ±0.04
   的折间标准差。多任务在本实验中"无伤害 + 消除特征穿越"，不承诺 ρ 提升。
3. **朴素联合 MSE 有害，必须做目标标准化**：原始分数目标（方差远小于
   one-hot 块）让共享干路被辅助块主导，ρ 崩到 0.45；分数目标 z-score 后
   恢复 0.611。若 Wave 1 之后重试联合形式，z-score 是硬前提。
4. **Wave 1 采用两阶段形式**：辅助标签由独立分类器预测、概率作为打分器
   特征。它可解释、可单测、辅助头可独立替换，且失败时退化为纯特征打分器。
5. **相似度特征是重要但非唯一信号**（−0.025ω 缺席时），S1.6 的 embedding
   fail-open 回落 LLM 代价可控。
6. **xhs 分层持续弱于 bilibili**（~0.45 vs ~0.58），与 spec S1.4 分平台
   门槛的担忧一致；xhs 侧互动特征缺失更多（无时长/播放量）是候选原因。

## 局限

n=686、单一用户、教师分带批内相对性（未做批内 pairwise 损失）、画像漂移
未建模（相似度取自评估时刻 audit，已是当时值）。数据量翻倍后应重跑本脚本
复核结论 2 与 3。
