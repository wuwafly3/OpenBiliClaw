# ML 准入推理（Gate）

> Wave 1 离线训练、运行时纯 numpy 前向。默认不改变 admission。
> **Gate ≠ ranker：** 本模块只做发现侧准入蒸馏。池内精排仍是
> `PoolCurator` 手工权重，学习排序见阶段 2 与
> [2026-08-21 分离 spec](../plans/2026-08-21-ml-gate-ranker-separation-spec.md)。

## 概述

`ml/` 把教师蒸馏出的浅 **gate**（准入分类器）接到 discovery 评估路径上。训练仍在 `[ml]` extra（scikit-learn）；daemon 只依赖 numpy。Gate 概率只挂内存字段 `ml_admission_*`，**不写** `relevance_score`。

当前特征版本 `admission-teacher-tags-v1`：确定性互动 / 平台 / 策略特征 + **廉价通道** `tag_channel_*`（`topic_group` / `style_key` / `temporal_class`）。教师 `topic_group` / `style_key` / `temporal_*` / `franchise_key` / 分数不得作为运行时特征。默认 artifact 为 `data/ml_artifacts/admission_teacher_v1.json`（`tags_source=teacher_oracle`，`row_filter=candidate_profile_digest`）。2026-08-20 快照子集 870 行、阈值 0.50：OOF AUC 0.741、agreement 0.700。`relevance_scorer` 默认仍为 `llm`。

生产 artifact 的下一步合同（尚未改训练入口）：只训 `evaluation_context_snapshots` 校验通过的行，并补画像相对特征；S1.5 一致率改为 0.95 × 快照自洽天花板（2026-08-20：0.756 → 0.718），Brier 不是合入门槛。Gate 教师画像从 2026-08-21 起不含 recent 层；旧快照若仍含 `recent_awareness` 等键，属另一套教师合同，不得与新切片混训成同一生产 artifact。

## 实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 共享特征编码 | ✅ | `features.py`：train 脚本与推理共用同一套列名 / vocab |
| numpy 前向 | ✅ | logistic + artifact 内 isotonic 表 + 决策阈 |
| 版本校验 / fail-open | ✅ | artifact 缺失、`feature_version` 不匹配、编码失败、缺廉价 tags → 不写 `relevance_score`，继续 LLM |
| 配置开关 | ✅ | `[discovery].relevance_scorer = llm \| shadow \| ml`，默认 `llm`。只动 gate，不改 curator / MMR |
| 跳过 y=0 完整评估 | ❌ | 改写后的 S1.5 未过；`ml` 仍观察性打分 |
| 教师自洽天花板 | 🧪 | `scripts/ml_teacher_self_consistency_probe.py`：同一 pinned 实例复测准入标签；只接受精确 `provider/model`，不混 adapter。有快照则按 digest 分组回放 labeling-time compact 画像。2026-08-20 快照回放 agreement **0.756** |
| 评估上下文快照 | ✅ | `discovery_candidates.profile_digest` / `negative_digest` + `evaluation_context_snapshots`；新教师标签可按打标时刻 prompt 回放 |
| Gate 评估不含 recent | ✅ | 教师 / digest 走 `compact_gate_evaluation_profile_summary`；recent 只留给推荐 compact / 未来 ranker |
| 学习排序 ranker | ❌ | 独立开关 `[recommendation].ranker`（尚未落地）；不以教师分为 y |

## 公共 API

```python
from openbiliclaw.ml.inference import AdmissionModel, resolve_model_path
from openbiliclaw.ml.features import feature_record_from_content, encode_features

model = AdmissionModel.load(path)
result = model.predict([feature_record_from_content(content, tag_source="cheap")])
```

`ContentDiscoveryEngine.score_admission_batch()` 在 `evaluate_content` / `evaluate_content_batch` 进入 LLM 之前调用。结果只挂在 `DiscoveredContent.ml_admission_*`（内存字段，不落库）。`shadow`/`ml` 在教师分返回后打 `ml-shadow disagreement` 日志（平台/策略/0-1，无标题）。

S1.5 的准入一致率对照教师噪声：`scripts/ml_teacher_self_consistency_probe.py` 只复测 `teacher_model` 与 pinned 实例完全一致的行（默认 `openai-4` → `openai/deepseek-v4-flash`），不把 `openai_compatible` 混进天花板。有 `evaluation_context_snapshots` 时按 `(profile_digest, negative_digest)` 分组回放，不把不同打标上下文塞进同一 batch；缺快照的历史行仍用当前画像（会低估天花板）。合入门槛是 **0.95 × 该天花板**，不是绝对 0.90。不改 admission，不跳过评估，不回写旧行 digest。写回见 [2026-08-19 探针记录](../plans/2026-08-19-ml-teacher-self-consistency-probe.md)、[评估上下文快照 spec](../plans/2026-08-19-ml-eval-context-snapshot-spec.md) 与 [gate/ranker 分离 spec](../plans/2026-08-21-ml-gate-ranker-separation-spec.md)。

`mypy` 对 `numpy` / `numpy.*` 使用 `follow_imports = skip`（含 stub）：numpy 2.5 stub 含 3.12 `type` 语句，而项目 mypy 目标是 3.11。
