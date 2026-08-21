"""Byte-equivalence + structural guards for the ``soul/profile_views`` façade.

Task 5 (Wave B) relocates the three profile serializers
(``build_profile_summary`` / ``compact_content_prompt_profile_summary`` /
``build_query_generation_profile_summary``) verbatim from
``discovery/strategies/_utils.py`` into ``soul/profile_views.py``, leaving
re-export stubs behind. This is a *mechanical move* — zero behaviour change.

Because ``_utils`` re-exports the moved names, the old import path and the new
module resolve to the *same function object* after the move, so comparing the
two live paths would be a tautology. Instead we freeze the pre-move output as
golden snapshot files (``tests/golden/profile_views/*.json``, generated from the
old ``_utils`` implementation before the relocation) and assert the new module
reproduces them byte-for-byte. That makes "output unchanged across the move" a
genuinely verified property, not a trivially-true assertion.

Three representative profiles exercise both branches of every serializer:

* ``young`` — small ``OnionProfile`` (~10 interests), no awareness/insights.
* ``mature`` — large ``OnionProfile`` (200+ specifics across 12 domains) with
  interest domains, recent awareness, active insights, MBTI, and active
  speculations.
* ``legacy_flat`` — the pre-onion flat ``SoulProfile`` shape (drives the
  category-reconstruction branch of ``_extract_interest_domains`` and the
  ``_raw_mbti`` branch of ``_summarize_mbti``).

The original move was verified against pre-move goldens. After that move, regenerate
only when the serializer contract changes intentionally, and review the resulting
snapshot diff rather than using regeneration to hide an accidental relocation change.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from openbiliclaw.soul.profile import (
    MBTI,
    AwarenessNote,
    ContextMode,
    CoreLayer,
    InsightHypothesis,
    InterestDomain,
    InterestLayer,
    InterestSpecific,
    InterestTag,
    MBTIDimension,
    OnionProfile,
    PreferenceLayer,
    RoleLayer,
    SoulProfile,
    StylePreference,
    SurfaceLayer,
    ValuesLayer,
)

_GOLDEN_DIR = Path(__file__).parent / "golden" / "profile_views"
_SRC_ROOT = Path(__file__).parent.parent / "src" / "openbiliclaw"

# The serializer names the façade owns. Their ``def`` must live only in
# ``soul/profile_views.py``.
_SERIALIZER_NAMES = (
    "build_profile_summary",
    "compact_content_prompt_profile_summary",
    "compact_gate_evaluation_profile_summary",
    "build_query_generation_profile_summary",
)


def _canonical(obj: object) -> str:
    """Deterministic JSON serialization (prompt-cache convention)."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Fixture profiles (deterministic — no randomness, no timestamps.now())
# ---------------------------------------------------------------------------


def young_profile() -> OnionProfile:
    """Small onion profile: ~10 interests, no awareness/insight churn."""
    return OnionProfile(
        personality_portrait="PORTRAIT_SENTINEL_young — 刚开始被理解的新用户。",
        core=CoreLayer(
            core_traits=["好奇", "随性"],
            deep_needs=["新鲜感"],
            mbti=MBTI(
                type="ENFP",
                confidence=0.4,
                dimensions={"EI": MBTIDimension(pole="E", strength=0.6)},
                inferred_from=["少量早期行为"],
            ),
        ),
        values_layer=ValuesLayer(values=["自由"], motivational_drivers=["尝试新事物"]),
        interest=InterestLayer(
            likes=[
                InterestDomain(
                    domain=f"领域-{i}",
                    weight=round(0.9 - i * 0.05, 3),
                    specifics=[
                        InterestSpecific(name=f"细分-{i}-{j}", weight=round(0.8 - j * 0.1, 3))
                        for j in range(2)
                    ],
                    first_seen="2026-06-01",
                    last_seen="2026-06-20",
                    source="behavior",
                )
                for i in range(5)
            ],
            dislikes=[
                InterestDomain(domain="标题党", weight=0.7),
            ],
            favorite_up_users=["某UP"],
        ),
        role=RoleLayer(life_stage="探索期", current_phase="刚接触"),
        surface=SurfaceLayer(
            cognitive_style=["跳跃式"],
            style=StylePreference(
                preferred_duration="short",
                preferred_pace="fast",
                quality_sensitivity=0.4,
                humor_preference=0.7,
                depth_preference=0.3,
            ),
            context=ContextMode(session_type="browsing"),
            exploration_openness=0.8,
        ),
        source_platform_mix={"bilibili": 1.0},
    )


def mature_profile() -> OnionProfile:
    """Large onion profile: 200+ specifics, awareness, insights, speculations."""
    profile = OnionProfile(
        personality_portrait="PORTRAIT_SENTINEL_mature — 长期被深度理解的用户。",
        core=CoreLayer(
            core_traits=[f"trait-{i}" for i in range(40)],
            deep_needs=[f"need-{i}" for i in range(40)],
            mbti=MBTI(
                type="INTJ",
                confidence=0.82,
                dimensions={
                    "EI": MBTIDimension(pole="I", strength=0.85),
                    "NS": MBTIDimension(pole="N", strength=0.7),
                },
                inferred_from=[f"信号-{i}" for i in range(40)],
            ),
        ),
        values_layer=ValuesLayer(
            values=[f"value-{i}" for i in range(40)],
            motivational_drivers=[f"driver-{i}" for i in range(40)],
        ),
        interest=InterestLayer(
            likes=[
                InterestDomain(
                    domain=f"领域-{i}",
                    weight=round(max(0.0, 0.99 - i * 0.05), 3),
                    specifics=[
                        InterestSpecific(
                            name=f"细分-{i}-{j}",
                            weight=round(max(0.0, 0.95 - j * 0.03), 3),
                        )
                        for j in range(20)
                    ],
                    first_seen="2026-01-01",
                    last_seen="2026-06-27",
                    source="behavior",
                )
                for i in range(12)
            ],
            dislikes=[
                InterestDomain(
                    domain=f"厌恶-{i}",
                    weight=round(max(0.0, 0.9 - i * 0.05), 3),
                    specifics=[InterestSpecific(name=f"低质-{i}", weight=0.7)],
                )
                for i in range(10)
            ],
            favorite_up_users=[f"UP-{i}" for i in range(8)],
        ),
        role=RoleLayer(life_stage="工作稳定期", current_phase="重新整理信息源"),
        surface=SurfaceLayer(
            cognitive_style=[f"style-{i}" for i in range(10)],
            style=StylePreference(
                preferred_duration="long",
                preferred_pace="moderate",
                quality_sensitivity=0.82,
                humor_preference=0.2,
                depth_preference=0.9,
            ),
            context=ContextMode(
                weekday_patterns="工作日晚间",
                weekend_patterns="周末深度",
                time_of_day_patterns="夜间为主",
                session_type="deep_dive",
            ),
            exploration_openness=0.66,
        ),
        source_platform_mix={"bilibili": 0.6, "youtube": 0.3, "xhs": 0.1},
        recent_awareness=[
            AwarenessNote(
                date=f"2026-06-{i + 1:02d}",
                observation=f"观察-{i} " + "细节" * 5,
                trend=f"趋势-{i}",
                emotion_guess=f"情绪-{i}",
            )
            for i in range(20)
        ],
        active_insights=[
            InsightHypothesis(
                hypothesis=f"假设-{i} " + "推理" * 5,
                evidence=[f"证据-{i}-{k}" for k in range(10)],
                confidence=round(0.6 + i * 0.01, 3),
                validated=bool(i % 2),
                created_at=f"2026-06-{i + 1:02d}T00:00:00",
            )
            for i in range(20)
        ],
    )
    # Active speculations are attached out-of-band on the live profile object.
    profile._active_speculations = [  # type: ignore[attr-defined]
        {"domain": f"猜测领域-{i}", "reason": f"因为行为信号-{i}"} for i in range(15)
    ]
    return profile


def legacy_flat_profile() -> SoulProfile:
    """Pre-onion flat profile: drives category-reconstruction + _raw_mbti paths."""
    profile = SoulProfile(
        personality_portrait="PORTRAIT_SENTINEL_legacy — 旧版扁平画像。",
        core_traits=["理性", "好奇", "克制"],
        cognitive_style=["结构化", "证据优先"],
        motivational_drivers=["理解底层", "降噪"],
        current_phase="整理信息源",
        values=["真实", "自主"],
        life_stage="工作稳定期",
        deep_needs=["确定性", "掌控感"],
        preferences=PreferenceLayer(
            interests=[
                InterestTag(
                    name=f"标签-{i}",
                    category=f"类别-{i % 6}",
                    weight=round(max(0.0, 1.0 - i * 0.02), 3),
                    first_seen="2026-01-01",
                    last_seen="2026-06-27",
                    source="behavior",
                )
                for i in range(30)
            ],
            style=StylePreference(
                preferred_duration="medium",
                preferred_pace="moderate",
                quality_sensitivity=0.6,
                humor_preference=0.5,
                depth_preference=0.6,
            ),
            context=ContextMode(session_type="deep_dive"),
            exploration_openness=0.55,
            disliked_topics=[f"不喜欢-{i}" for i in range(12)],
            favorite_up_users=["老UP"],
            source_platform_mix={"bilibili": 0.8, "youtube": 0.2},
        ),
        recent_awareness=[
            AwarenessNote(
                date=f"2026-06-{i + 1:02d}",
                observation=f"旧观察-{i}",
                trend=f"旧趋势-{i}",
                emotion_guess=f"旧情绪-{i}",
            )
            for i in range(8)
        ],
        active_insights=[
            InsightHypothesis(
                hypothesis=f"旧假设-{i}",
                evidence=[f"旧证据-{i}"],
                confidence=0.7,
                validated=bool(i % 2),
            )
            for i in range(6)
        ],
    )
    profile._raw_mbti = {  # type: ignore[attr-defined]
        "type": "INTP",
        "confidence": 0.6,
        "dimensions": {"EI": {"pole": "I", "strength": 0.7}},
        "inferred_from": ["历史观看"],
    }
    return profile


_FIXTURES: dict[str, object] = {
    "young": young_profile,
    "mature": mature_profile,
    "legacy_flat": legacy_flat_profile,
}


def _render_all(module: object) -> dict[str, str]:
    """Render every (serializer × fixture) pair to canonical JSON via *module*.

    ``module`` must expose the three serializer names. Returns a flat map of
    ``"{serializer}__{fixture}" -> canonical json`` (9 entries).
    """
    build_profile_summary = module.build_profile_summary  # type: ignore[attr-defined]
    compact = module.compact_content_prompt_profile_summary  # type: ignore[attr-defined]
    build_query = module.build_query_generation_profile_summary  # type: ignore[attr-defined]

    out: dict[str, str] = {}
    for fixture_name, factory in _FIXTURES.items():
        profile = factory()  # type: ignore[operator]
        out[f"build_profile_summary__{fixture_name}"] = _canonical(build_profile_summary(profile))
        out[f"compact_content_prompt_profile_summary__{fixture_name}"] = _canonical(
            compact(build_profile_summary(profile))
        )
        out[f"build_query_generation_profile_summary__{fixture_name}"] = _canonical(
            build_query(profile)
        )
    return out


# ---------------------------------------------------------------------------
# Byte-equivalence: new module vs frozen pre-move golden (9 comparisons)
# ---------------------------------------------------------------------------


def test_profile_views_match_pre_move_golden() -> None:
    """Every serializer × fixture reproduces the pre-move golden byte-for-byte."""
    from openbiliclaw.soul import profile_views

    rendered = _render_all(profile_views)
    assert len(rendered) == 9

    for key, actual in sorted(rendered.items()):
        golden_path = _GOLDEN_DIR / f"{key}.json"
        assert golden_path.exists(), f"missing golden snapshot: {golden_path}"
        expected = golden_path.read_text(encoding="utf-8")
        assert actual == expected, f"byte mismatch for {key}"


def _render_speculation() -> dict[str, str]:
    """Render the ``speculation`` view for every fixture (Task 7).

    Keyed ``speculation__{fixture}`` → the string block the speculator /
    avoidance-speculator prompts embed. The frozen goldens were generated from
    the pre-move ``to_llm_context(include_portrait=False)`` call, so matching
    them proves the façade view reproduces the fork output section-for-section.
    """
    from openbiliclaw.soul import profile_views

    out: dict[str, str] = {}
    for fixture_name, factory in _FIXTURES.items():
        profile = factory()  # type: ignore[operator]
        out[f"speculation__{fixture_name}"] = profile_views.speculation(profile)
    return out


def test_speculation_view_matches_pre_move_golden() -> None:
    """The ``speculation`` view reproduces the pre-move prompt block byte-for-byte.

    Also pins the delegation contract: the façade view is section-identical to
    the profile's own ``to_llm_context(include_portrait=False)`` renderer that
    the two speculator call sites used before Task 7.
    """
    rendered = _render_speculation()
    assert len(rendered) == len(_FIXTURES)

    for fixture_name, factory in _FIXTURES.items():
        key = f"speculation__{fixture_name}"
        golden_path = _GOLDEN_DIR / f"{key}.txt"
        assert golden_path.exists(), f"missing golden snapshot: {golden_path}"
        expected = golden_path.read_text(encoding="utf-8")
        assert rendered[key] == expected, f"byte mismatch for {key}"
        # Delegation is exact — no re-rendering drift vs the profile's own method.
        profile = factory()  # type: ignore[operator]
        assert rendered[key] == profile.to_llm_context(include_portrait=False)


def test_utils_reexports_are_the_facade_objects() -> None:
    """The legacy ``_utils`` names must resolve to the façade's objects."""
    from openbiliclaw.discovery.strategies import _utils
    from openbiliclaw.soul import profile_views

    for name in _SERIALIZER_NAMES:
        assert getattr(_utils, name) is getattr(profile_views, name), name


def test_compact_gate_evaluation_profile_summary_drops_recent_keeps_tags() -> None:
    from openbiliclaw.soul.profile_views import (
        build_profile_summary,
        compact_content_prompt_profile_summary,
        compact_gate_evaluation_profile_summary,
    )

    summary = build_profile_summary(mature_profile())
    ranked = compact_content_prompt_profile_summary(summary)
    gate = compact_gate_evaluation_profile_summary(summary)

    assert ranked["recent_awareness"]
    assert ranked["active_insights"]
    assert ranked["speculative_interests"]
    for key in ("recent_awareness", "active_insights", "speculative_interests"):
        assert key not in gate
    assert gate["interests"] == ranked["interests"]
    assert gate["disliked_topics"] == ranked["disliked_topics"]
    assert gate["style"] == ranked["style"]


# ---------------------------------------------------------------------------
# Structural guard (invariant V1): serializers defined only in profile_views
# ---------------------------------------------------------------------------


def _module_defines(path: Path, names: set[str]) -> set[str]:
    """Return which of *names* are top-level ``def``s in the file at *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found.add(node.name)
    return found


def test_serializers_defined_only_in_profile_views() -> None:
    """The serializer ``def``s appear in exactly one file: profile_views."""
    names = set(_SERIALIZER_NAMES)
    definers: dict[str, list[str]] = {name: [] for name in names}
    for path in _SRC_ROOT.rglob("*.py"):
        for name in _module_defines(path, names):
            definers[name].append(path.relative_to(_SRC_ROOT).as_posix())

    for name, files in definers.items():
        assert files == ["soul/profile_views.py"], f"{name} defined in {files}"


def test_content_pipeline_imports_from_facade_or_reexport() -> None:
    """Discovery/recommendation/runtime/sources reference the serializers only
    through ``profile_views`` or the ``_utils`` re-export — never a private fork.
    """
    watched_dirs = ("discovery", "recommendation", "runtime", "sources")
    allowed_import_roots = (
        "openbiliclaw.soul.profile_views",
        "openbiliclaw.discovery.strategies._utils",
    )
    names = set(_SERIALIZER_NAMES)

    for top in watched_dirs:
        for path in (_SRC_ROOT / top).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Names this module imports (module-level or function-level).
            imported: set[str] = set()
            import_sources: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        if alias.name in names:
                            imported.add(alias.name)
                            import_sources[alias.name] = node.module
            for name in imported:
                source = import_sources[name]
                assert (
                    source in allowed_import_roots
                ), f"{path.relative_to(_SRC_ROOT)} imports {name} from {source}"


if __name__ == "__main__":  # pragma: no cover — golden generation helper
    import sys

    if "--generate" in sys.argv:
        # Generate goldens from the current canonical implementation. This was
        # originally run against pre-move ``_utils``; after the relocation it is
        # reserved for reviewed, intentional serializer-contract changes.
        from openbiliclaw.discovery.strategies import _utils

        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for _key, _payload in _render_all(_utils).items():
            (_GOLDEN_DIR / f"{_key}.json").write_text(_payload, encoding="utf-8")
        # Speculation goldens: the pre-move source is the profile's own renderer
        # (the view did not exist yet), so freeze that output directly.
        for _name, _factory in _FIXTURES.items():
            _block = _factory().to_llm_context(include_portrait=False)  # type: ignore[operator]
            (_GOLDEN_DIR / f"speculation__{_name}.txt").write_text(_block, encoding="utf-8")
        print(f"wrote {len(_FIXTURES) * 4} goldens to {_GOLDEN_DIR}")
