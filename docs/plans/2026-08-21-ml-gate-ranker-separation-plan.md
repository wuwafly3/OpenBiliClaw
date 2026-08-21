# Gate / Ranker 分离 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans (execute this plan task-by-task).
> **Spec:** [`2026-08-21-ml-gate-ranker-separation-spec.md`](./2026-08-21-ml-gate-ranker-separation-spec.md)
> **Parent:** [`2026-08-15-ml-ranking-spec.md`](./2026-08-15-ml-ranking-spec.md) /
> [`2026-08-15-ml-ranking-plan.md`](./2026-08-15-ml-ranking-plan.md)
> **Status:** draft; Wave A = 文档合同 + 快照训练过滤 + 画像相对特征；默认
> `relevance_scorer` 仍为 `llm`
> **Execution order:** Task 1 → 2 → 3 → 4（Wave A）。Task 5（ranker）可停。
> **Tech:** Python 3.11；`uv run pytest tests/test_train_relevance_model.py
> tests/test_eval_context.py tests/test_ml_teacher_self_consistency_probe.py -q`；
> `uv run ruff check src tests scripts`；`uv run mypy src`

**Invariants that MUST hold — re-read before each task:**

- Gate 不写 `relevance_score`；教师分只来自 LLM 路径。
- Gate 教师画像不含 recent 层；ranker / 推荐表达可以含。
- Ranker 不以教师分为 y；正样本 < 200 不得进 `ranker=shadow`。
- `[discovery].relevance_scorer` 与 `[recommendation].ranker` 独立。
- 生产 gate 只训 `digests_match()` 的 snapshot 行；推理用 live compact+负例。
- 阶段 1 只标定 C1。C2/C3/C6 留在教师分上。C4/C5 留给 ranker。
- S1.5 一致率是 0.95 × 快照自洽天花板，不是绝对 0.90。Brier 不是合入门槛。

### Task 1: 父 spec/plan 冲突条款改指向本修订

**Files:** Add/modify/test
`docs/plans/2026-08-15-ml-ranking-spec.md`,
`docs/plans/2026-08-15-ml-ranking-plan.md`,
`docs/modules/ml.md`,
`docs/changelog.md`,
`docs/modules/config.md`（仅说明 scorer ≠ ranker，不改字段名）。

**Interfaces:** Consumes: 本 spec 对照表。Produces: 执行者读 8-15 会落到 8-21。

**Steps:**

- [ ] Write 本对文档（已完成则核对 Status / 对照表仍与代码现状一致）。
- [ ] 在 8-15 spec 顶部 Status 写明：阶段 1/2 验收以 8-21 为准；S1.5 Brier、
      S1.6 C2–C6 作废。
- [ ] 在 8-15 plan Wave 0 第 5 步改为 Wave 2 前置；Wave 1 第 7–8 步去掉
      C1–C7 整包重标定与 Brier 合入。
- [ ] 更新 `docs/modules/ml.md` 概述：当前模型是观察性 **gate**，不是 ranker。
- [ ] `docs/changelog.md` 未发布块加一条分离合同。
- [ ] Run 不涉及代码；人工确认 grep 父 plan Wave 1 不再出现
      「C1–C7 全部 7 个常数」作为合入门。

**Acceptance:**

- Numeric gate: 无运行指标；冲突条款 0 条仍被写成 Wave 1 合入条件。
- Reproduce with 阅读 8-15 §阶段 1 与 Wave 1 门；记录指向 8-21。

### Task 2: 生产训练只接受验证快照行

**Files:** Add/modify/test
`scripts/train_relevance_model.py`,
`tests/test_train_relevance_model.py`,
`docs/modules/ml.md`.

**Interfaces:** Consumes: `discovery_candidates` digest 列 +
`Database.get_evaluation_context_snapshot`。Produces: artifact metadata
`row_filter=verified_snapshot`（比 `candidate_profile_digest` 更严）。

**Steps:**

- [ ] Write one focused failing test：digest 非空但 snapshot 缺失 / 
      `digests_match() is False` 的行在 `--require-snapshot` 下被丢弃。
- [ ] Run `uv run pytest tests/test_train_relevance_model.py -q` and confirm
      FAIL for the intended missing behavior.
- [ ] Add `--require-snapshot`（可与现有 `--require-profile-digest` 并存；
      生产默认走 snapshot）。空 digest 与校验失败计入 stats，不进矩阵。
- [ ] Rerun the focused test and confirm PASS with no warnings.
- [ ] Run `uv run ruff check scripts/train_relevance_model.py
      tests/test_train_relevance_model.py` 与 touched pytest。

**Acceptance:**

- Numeric gate: 丢弃行全部属于「无快照 / digest 不匹配 / 非教师白名单」；
  0 行校验失败被写入 artifact。
- Reproduce with `uv run --extra ml python scripts/train_relevance_model.py
  --require-snapshot --dry-run`（或等价只读路径）；记录
  `row_filter` 与行数。不改 `relevance_scorer`。

### Task 3: 画像相对特征（进 `ml` 前 MUST）

**Files:** Add/modify/test
`src/openbiliclaw/ml/features.py`,
`scripts/train_relevance_model.py`,
`tests/test_train_relevance_model.py`,
`tests/test_ml_features.py`（若无则新建）,
`docs/modules/ml.md`.

**Interfaces:** Consumes: snapshot `profile_summary` + `negative_examples` +
候选文本 / cheap tags。Produces: 新 `FEATURE_VERSION`；live 路径用同一纯函数
吃当前 compact+负例。

**Steps:**

- [ ] Write one focused failing test：同一候选、两份 snapshot（dislike /
      负例不同）编码后向量不同；教师 tags / `franchise_key` / 分数仍不在
      live 特征名里。
- [ ] Run the focused test and confirm FAIL for the missing relative features.
- [ ] Add 最少相对量：t0/live compact 可嵌入文本 vs 候选余弦（或复用打标时刻
      `sim` 并标明来源）、dislike 命中、负例标题重叠、style 偏好 vs cheap
      `style_key`。Bump `FEATURE_VERSION`。旧 `admission-teacher-tags-v1`
      仅对照。
- [ ] Rerun focused test PASS. Run ruff + mypy on `src/openbiliclaw/ml`.
- [ ] 不打开 `relevance_scorer=ml`。对照训练可记 OOF 测量，不是合入门槛。

**Acceptance:**

- Numeric gate: live 特征名集合与教师 `topic_group` / `style_key` /
  `franchise` / `llm_score` 交集为空；相对特征至少 1 个随 snapshot 变化。
- Reproduce with focused pytest；记录新 `FEATURE_VERSION` 字符串。

### Task 4: 评估脚本按相对 S1.5 报告（Brier 降级）

**Files:** Add/modify/test
`scripts/evaluate_relevance_distillation.py`（若尚未存在则按 8-15 Wave 1
第 7 步新建）,
对应 `tests/`,
`docs/plans/2026-08-19-ml-teacher-self-consistency-probe.md`（交叉引用天花板）.

**Interfaces:** Consumes: holdout 预测 + 同期
`teacher_self_consistency_*.json` 的 snapshot agreement。Produces: 一致率
门槛 = `0.95 * ceiling`；Brier 打印但不 fail。

**Steps:**

- [ ] Write one focused failing test：ceiling=0.756 时 agreement 0.718 过线、
      0.717 不过；Brier 0.20 不单独失败。
- [ ] Run focused test FAIL.
- [ ] Implement 门槛计算与分平台 FPR/FNR。n<100 标记 `insufficient_holdout`，
      不得宣称 S1.5 通过。
- [ ] Rerun PASS. 不改 admission 运行时。

**Acceptance:**

- Numeric gate: 门槛公式锁死为 `0.95 × snapshot self-consistency agreement`；
  2026-08-20 天花板代入为 0.718。
- Reproduce with focused pytest。

### Task 5: Ranker（可停；不在 Wave A）

**Files:** Add/modify/test `src/openbiliclaw/ml/ranker.py`（新）,
`recommendation/curator.py` 装配，`tests/`，配置
`[recommendation].ranker`。

**Interfaces:** Consumes: `engagement_label` + 现有疲劳/单调统计量。
Produces: 替换 `score_candidates` 的分；默认 `weights`。

**Steps:**

- [ ] 正样本 < 200 时测试拒绝进入 `shadow`。
- [ ] 训练脚本拒绝教师分当 y。
- [ ] NDCG@10 bootstrap 区间不含 0 才允许 `ml`。
- [ ] gate 开关保持 `llm`；打开 ranker 不得改 `relevance_score` 写入。

**Acceptance:**

- Numeric gate: 父 spec S2.4 / S2.5。
- Reproduce with ranker 评估脚本（落地时写命令）。

## Verification after merge

Wave A 合入后：默认路径仍是 LLM 教师打分。观察
`evaluation_context_snapshots` 覆盖率与 `--require-snapshot` 训练集规模；
覆盖不足时不训生产 artifact。回滚：删除新特征版本 artifact，训练入口回到
文档所述对照模式。Ranker 未上线，无 serve 回滚。

## Explicitly out of scope

- 把 gate 概率写入 `relevance_score` 或 curator 线性项
- 用教师分训 ranker
- 阶段 1 重标定 C2–C7
- 绝对 agreement 0.90
- 用 300 条曝光负样本阻塞 gate 训练
- 打开 live `ml` 跳过 y=0 评估
- 配置键改名
