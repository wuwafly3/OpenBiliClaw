# Gate 合同重打标 Spec — 旧快照去 recent 后重评教师分

**Created:** 2026-08-23
**Status:** draft
**Parent:** [`2026-08-21-ml-gate-ranker-separation-spec.md`](./2026-08-21-ml-gate-ranker-separation-spec.md)
**Also amends:** [`2026-08-19-ml-eval-context-snapshot-spec.md`](./2026-08-19-ml-eval-context-snapshot-spec.md)
**Scope:** 把 8-20 含 recent 的评估快照改写成新 gate 合同（无 recent、无页面壳标题），用 pinned 教师重打分；新标签另存，旧行分数 / digest 不覆盖。
**Out of scope:** 打开 `relevance_scorer=ml`、Task 2 `--require-snapshot` 训练入口、画像相对特征、ranker、explore 去教师、插件 / desktop / mobile / CLI 推荐 UI。

## Goal

新合同（`compact_gate_evaluation_profile_summary`，无 recent）下，生产可训行目前只有今天打的约 229 条，且缺 B 站 / 小红书。8-20 有 **918** 条 pinned 教师 + 校验快照行，但快照仍带 `recent_awareness` / `active_insights` / `speculative_interests`，以及干杯壳标题。

可验证结果：

- 纯函数把旧快照改写成新合同：丢掉三 recent 键，丢掉 `is_shell_negative_title` 命中的负例，重算 digest。
- 用改写后的快照绑着这 918 个候选，再跑 pinned `openai-4`（`openai/deepseek-v4-flash`）的 `evaluate_batch`。
- 新分数写入版本化 JSONL；新快照 upsert 进 `evaluation_context_snapshots`。**不** UPDATE `discovery_candidates.relevance_score` / `llm_score_raw` / digest。
- `--dry-run` 在不调 LLM 的情况下报告：改写后组数、行数、分平台。2026-08-23 实测口径：918 行、17 个新 digest 组、B 站 571 / 小红书 156 / YouTube 106 / Twitter 85。

验证：

```text
uv run pytest tests/test_eval_context.py tests/test_ml_gate_contract_relabel.py tests/test_negative_exemplars.py -q
uv run python scripts/ml_gate_contract_relabel.py --dry-run
```

## Design invariants (MUST hold in every phase)

1. **不覆盖旧教师行：** 脚本不得 `UPDATE discovery_candidates` 的 `relevance_score`、`llm_score_raw`、`score_source`、`teacher_model`、`profile_digest`、`negative_digest`。8-20 旧分仍是 0.756 快照自洽天花板的证据。验证：单测断言无这类 SQL；dry-run / 实跑前后抽样旧行分数不变。
2. **新标签必须是新 prompt 下的教师分：** 不得把旧 `llm_score_raw` 改 digest 后当新 y。改写只动画像切片和负例列表。验证：输出 JSONL 同时保留 `old_llm_score_raw` 与 `new_llm_score_raw`。
3. **改写是纯函数：** `rewrite_snapshot_to_gate_contract` 吃一份校验通过的 `EvaluationContextSnapshot`，产出新 snapshot：`profile_summary` 经 `compact_gate_evaluation_profile_summary`（无三 recent 键）；`negative_examples` 丢掉壳标题、保持相对顺序、不补到 16；`recall_pool` 原样；新 digest `digests_match()`。验证：同一输入两次字节级相等；含 recent 的输入与去 recent 后 digest 不同。
4. **壳标题过滤与 live 装配同一函数：** 用 `is_shell_negative_title`。名单只列完整网站标题 `哔哩哔哩 (゜-゜)つロ 干杯~-bilibili`；干杯短标题靠去掉尾部站点品牌匹配。`ChatGLM`、带 `_哔哩哔哩_bilibili` 的稿件名不是壳。不删 `events`。
5. **旧快照行保留：** upsert 新 `(profile_digest, negative_digest)` 不得删除 8-20 含 recent 的主键。验证：实跑后旧 digest 对仍能 `get_evaluation_context_snapshot`。
6. **身份与分组沿用自洽探针：** 只重评 `teacher_model == openai/deepseek-v4-flash`。`openai_compatible` 不混。按**新** digest 对分组，不同对不进同一 `evaluate_batch`。`eval_prefilter_mode=off`，跳过 recently-viewed。配额 / 身份失败不静默换实例。
7. **Fail-open 与探针一致：** 硬配额停在已完成 chunk，已写出的 JSONL 保留；可 `--resume` 跳过已有 `candidate_id`。不把半截身份错的 chunk 写入标签。
8. **默认 scorer 仍为 llm：** 本切片不改 `[discovery].relevance_scorer`。

## Current diagnosis

### D1. 新合同可训行不够分平台

2026-08-23 只读盘点：教师白名单 7311；pinned 6524；校验快照 1065。其中旧合同（含 recent）918 行 / 8-20；新合同 229 行 / 8-23，平台只有 twitter + youtube。S1.5 要 holdout n≥100 且分平台过线。缺 B 站 / 小红书就不能训生产 artifact。

### D2. 旧分数不能改 digest 混进新 `FEATURE_VERSION`

分离 spec 不变量 8：历史快照若仍带 recent 键，不得与新切片混成同一生产 artifact。8-20 快照回放 agreement 0.756 对的是**含 recent 的原 prompt**。去 recent 后尺子变了，y 会变。

### D3. 现成自洽探针不能当重打标器

`scripts/ml_teacher_self_consistency_probe.py` 默认抽 90 条、不落新标签、不改写快照。它是天花板测量，不是换合同后的重新打标。

### D4. 8-20 负例含页面壳

24 份旧快照每份有干杯壳标题。改写时必须丢掉，否则新合同负例仍在教站点 chrome。丢掉后不从事件库补位（那会混入打标时刻之后的负例）。

## Priority classification

| Phase | Content | Tier | Why |
| --- | --- | --- | --- |
| 0 | 本文档 | **MUST** | 执行者不得把旧分当新标签 |
| 1 | 壳标题过滤（若本分支尚未落地） | **MUST** | 改写与 live 装配共用 |
| 2 | `rewrite_snapshot_to_gate_contract` | **MUST** | 否则脚本会手写切片 |
| 3 | 重打标脚本 + JSONL | **MUST** | 918 行的唯一放大手段 |
| 4 | 实跑 pinned 教师 | **MUST** | dry-run 不够产生 y |
| 5 | `--require-snapshot` 训练入口 | 8-21 Task 2 | 本切片不改训练脚本 |

Wave A 数据准备 = Phase 0–4。可在 Task 2 训练过滤之前停止。

## Phase designs

### Phase 0 — 合同

- 新合同训练集 = 今天已无 recent 的快照行 ∪ 本脚本 JSONL 中的重打标行。
- 旧合同行留在 `discovery_candidates`，只作对照 / 天花板回放。

### Phase 1 — 壳标题

与分离 spec D6 相同。本分支 HEAD 若还没有 `is_shell_negative_title`，在本切片落地，供 live 装配和改写共用。

### Phase 2 — 改写

```
rewrite(old) :=
  summary' = compact_gate_evaluation_profile_summary(old.profile_summary)
  negatives' = [n in old.negative_examples if not is_shell_negative_title(n.title)]
  recall' = old.recall_pool
  digest' = hash(summary', recall') / hash(negatives')
```

已是新合同且无壳标题的快照：改写后 digest 不变，不进入重评队列。

### Phase 3 — 脚本

`scripts/ml_gate_contract_relabel.py`

- `--dry-run`：只打印改写统计与分组，不调 LLM，不写库。
- 默认读 live `config.toml` + `data/openbiliclaw.db`。
- 实跑：按新 digest 组调用 `evaluate_content_batch`（复用探针的 pin / 429 / 身份守卫）。
- 写出 `data/ml_artifacts/gate_contract_relabel_<ts>.jsonl` + `.meta.json`。
- 成功 chunk 后 upsert 新 snapshot。
- `--resume PATH` 跳过 JSONL 里已有的 `candidate_id`。
- `--limit N` 仅冒烟。

### Phase 4 — 实跑门槛

| 指标 | 门槛 |
| --- | --- |
| dry-run 行数 | pinned + 旧合同快照 = **918**（相对 2026-08-23 库；库增长则按同一过滤重报） |
| 新 digest 组 | ≥ 10（实测 17） |
| 分平台 | B 站、小红书、YouTube、Twitter 均 > 0 |
| 实跑写入 | `new_teacher_model == openai/deepseek-v4-flash`；身份不匹配的 chunk 丢弃 |
| 旧行 | 抽样 `llm_score_raw` 与跑前一致 |
| 成本 | 按 8-20 探针单价约 ¥50 / 918 行；记入 meta，不是合入门槛 |

n<100 的实跑不得宣称 S1.5；本切片只生产训练集。

## Expected impact

| Lever | Measured effect |
| --- | --- |
| 重打标 918 行 | 新合同可训行 229 → 约 1147，补上 B 站 / 小红书 |
| 去 recent 合并组 | 24 份旧快照塌成更少稳定画像组（实测 17 组有行） |
| 不覆盖旧分 | 0.756 天花板回放仍可对原 snapshot 跑 |

## Documentation obligations

- 本文 + plan
- `docs/modules/ml.md` — 新合同数据来自 live 新行 ∪ 重打标 JSONL
- `docs/changelog.md` 未发布块一条
- `docs/modules/soul.md` — 若本切片落地壳标题过滤
- 不改架构图 / README highlights（未上线推理）
