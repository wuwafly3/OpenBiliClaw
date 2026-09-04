# Gate 合同重打标 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans (execute this plan task-by-task).
> **Spec:** [`2026-08-23-ml-gate-contract-relabel-spec.md`](./2026-08-23-ml-gate-contract-relabel-spec.md)
> **Parent:** [`2026-08-21-ml-gate-ranker-separation-spec.md`](./2026-08-21-ml-gate-ranker-separation-spec.md)
> **Status:** draft
> **Execution order:** Task 1（文档，本提交）→ Task 2（壳标题，若 HEAD 没有）→ Task 3（改写纯函数）→ Task 4（脚本 + dry-run）→ Task 5（实跑 LLM，需用户确认费用）。
> **Tech:** Python 3.11；`uv run pytest tests/test_eval_context.py tests/test_ml_gate_contract_relabel.py tests/test_negative_exemplars.py tests/test_ml_teacher_self_consistency_probe.py -q`；`uv run ruff check src tests scripts`；`uv run mypy src`

**Invariants that MUST hold — re-read before each task:**

- 不 `UPDATE discovery_candidates` 的分数 / digest。
- 新 y 必须来自新 prompt 下的 pinned 教师分，不得复用旧 `llm_score_raw`。
- 改写是纯函数：去 recent + 丢壳标题 + 重算 digest。
- 壳标题与 live 装配共用 `is_shell_negative_title`；`ChatGLM` 不是壳。
- 旧含 recent 的 snapshot 主键保留。
- 按新 digest 对分组；不混 `openai_compatible`。
- 默认 `relevance_scorer` 仍为 `llm`。

### Task 1: 文档合同

**Files:** Add
`docs/plans/2026-08-23-ml-gate-contract-relabel-spec.md`,
`docs/plans/2026-08-23-ml-gate-contract-relabel-plan.md`.

**Interfaces:** Consumes: 8-21 分离 spec 不变量 4/5/8 与 8-23 只读盘点。Produces: 执行者不得把旧分当新标签。

**Steps:**

- [x] Write 本对文档。
- [ ] Commit `docs: add gate-contract relabel spec and plan`（docs-first，不含代码）。

**Acceptance:**

- Numeric gate: 无运行指标。
- Reproduce with 阅读 spec 不变量 1–2。

### Task 2: 壳标题过滤（本分支 HEAD 若未落地）

**Files:** Add/modify/test
`src/openbiliclaw/soul/negative_exemplars.py`,
`tests/test_negative_exemplars.py`,
`docs/modules/soul.md`.

**Interfaces:** Consumes: 负例事件标题。Produces: `is_shell_negative_title`；`recent_negative_exemplars` 丢壳。

**Steps:**

- [ ] 若 `is_shell_negative_title` 已存在则跳过。
- [ ] Write focused failing test：干杯完整标题与短标题被丢，`ChatGLM` 保留。
- [ ] 实现黑名单 + 去掉尾部 ASCII 站点品牌的指纹。
- [ ] Rerun PASS. 不改教师 system prompt。

**Acceptance:**

- Numeric gate: 上述两例 0 条进入返回列表；至少 1 条真实内容标题仍在。
- Reproduce with `uv run pytest tests/test_negative_exemplars.py -q`.

### Task 3: 快照改写纯函数

**Files:** Add/modify/test
`src/openbiliclaw/discovery/eval_context.py`,
`tests/test_eval_context.py`.

**Interfaces:** Consumes: 校验通过的旧 `EvaluationContextSnapshot`。Produces: 新合同 snapshot；`had_recent` / `dropped_shell` 统计可附在返回值旁。

**Steps:**

- [ ] Write failing test：含 recent + 干杯壳的快照，改写后无 recent 键、无壳标题、digest 变且 `digests_match()`；`ChatGLM` 保留；同一输入两次相等。
- [ ] Add `rewrite_snapshot_to_gate_contract`.
- [ ] Rerun PASS.

**Acceptance:**

- Numeric gate: recent 键交集为空；壳标题 0；内容标题 ≥1。
- Reproduce with focused pytest.

### Task 4: 重打标脚本（先 dry-run）

**Files:** Add/modify/test
`scripts/ml_gate_contract_relabel.py`,
`tests/test_ml_gate_contract_relabel.py`,
`docs/modules/ml.md`,
`docs/changelog.md`.

**Interfaces:** Consumes: live DB 教师白名单 + snapshots。Produces: JSONL + meta；可选 upsert 新 snapshot。不写候选分数。

**Steps:**

- [ ] Write failing tests：过滤只收 pinned + 旧合同快照；dry-run 不调用 eval；输出行含 old/new 分数字段；禁止 UPDATE 候选分数列。
- [ ] Implement 脚本：复用探针的 pin / 身份守卫 / chunk 429；按新 digest 分组。
- [ ] `uv run python scripts/ml_gate_contract_relabel.py --dry-run` 对照 spec：约 918 行、≥10 组、四平台均有。
- [ ] 文档：ml.md + changelog。

**Acceptance:**

- Numeric gate: dry-run kept 行 = pinned 且旧快照含 recent 的教师行；0 次 LLM。
- Reproduce with `--dry-run`.

### Task 5: 实跑（单独一步，先 dry-run 通过）

**Files:** `data/ml_artifacts/gate_contract_relabel_*.jsonl`（gitignored）。

**Steps:**

- [ ] 用户确认费用后去掉 `--dry-run`。
- [ ] `--resume` 若中断。
- [ ] 核对 meta：写入行的 `new_teacher_model` 全是 pinned；抽样旧 `discovery_candidates.llm_score_raw` 未变；旧 snapshot 主键仍在。

**Acceptance:**

- Numeric gate: 成功写入 ≥100 才可用于后续训练实验；全量目标 918。身份错的 chunk 不入库。
- Reproduce with meta.json + 只读 SQL。

## Verification after merge

默认路径仍是 LLM 教师。观察 JSONL 行数与新 snapshot 对。回滚：删 JSONL、不删旧 snapshot。不改 `relevance_scorer`。

## Explicitly out of scope

- 覆盖 `discovery_candidates` 旧分
- 把旧分改 digest 当新 y
- 打开 live `ml`
- Task 2 训练入口 / Task 3 相对特征（8-21 plan）
- Ranker / explore 去教师
