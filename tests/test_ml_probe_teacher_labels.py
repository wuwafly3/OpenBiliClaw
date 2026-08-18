"""Guards that offline distill probes stay on the export teacher-label policy.

The probe scripts import numpy/scikit-learn at module load, which is not a
runtime or pytest dependency. These tests therefore assert the call-site
contract against source rather than executing training.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_PROBE_SCRIPTS = (
    "ml_multitask_distill_probe.py",
    "ml_pairwise_distill_probe.py",
    "ml_tags_ablation_probe.py",
)


def test_distill_probes_delegate_teacher_labels_to_export_policy() -> None:
    for name in _PROBE_SCRIPTS:
        text = (_SCRIPTS / name).read_text(encoding="utf-8")
        assert "from export_ranking_dataset import" in text, name
        assert "resolve_teacher_label(" in text, name
        assert "teacher_label_select_columns(" in text, name
        assert "candidate_row_for_label(" in text, name
        assert "if persisted > 0.0:" not in text, name


def test_tags_probe_uses_exact_explore_admission_floor() -> None:
    text = (_SCRIPTS / "ml_tags_ablation_probe.py").read_text(encoding="utf-8")
    assert "effective_admission_threshold(" in text
    assert '"explore" in r["strategy"]' not in text
    assert '"explore" in r[' not in text
