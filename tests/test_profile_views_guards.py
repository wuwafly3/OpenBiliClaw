"""Guard tests for the profile→LLM serialization boundary.

These pin two standing invariants that no test previously covered:

1. **Portrait boundary** — the free-form ``personality_portrait`` narrative must
   never leak into a content-pipeline serializer. The designed dict
   serializers (``build_profile_summary`` / ``compact_content_prompt_profile_summary``
   / ``compact_gate_evaluation_profile_summary`` /
   ``build_query_generation_profile_summary``) and the string-shaped
   ``OnionProfile.to_llm_context(include_portrait=False)`` fork are all fed a
   sentinel portrait and asserted to exclude it.
2. **Determinism** — each serializer is a pure function of the profile: two calls
   on equal input serialize byte-identically under the canonical
   ``json.dumps(..., ensure_ascii=False, indent=2, sort_keys=True)``.

The tests pin *already-correct* behavior (they are a regression net, not a
red-to-green driver). A mutation that reintroduces the portrait — e.g. adding
``"personality_portrait": profile.personality_portrait`` to
``build_profile_summary`` — must turn the (a) assertions red.
"""

from __future__ import annotations

import json

from openbiliclaw.discovery.strategies._utils import (
    build_profile_summary,
    build_query_generation_profile_summary,
    compact_content_prompt_profile_summary,
    compact_gate_evaluation_profile_summary,
)
from openbiliclaw.soul.profile import (
    AwarenessNote,
    CoreLayer,
    InsightHypothesis,
    InterestDomain,
    InterestLayer,
    InterestSpecific,
    InterestTag,
    OnionProfile,
    PreferenceLayer,
    RoleLayer,
    SoulProfile,
    SurfaceLayer,
    ValuesLayer,
)

_PORTRAIT_SENTINEL = "PORTRAIT_SENTINEL_XYZ"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _soul_profile() -> SoulProfile:
    """A populated ``SoulProfile`` carrying the sentinel portrait."""
    return SoulProfile(
        personality_portrait=_PORTRAIT_SENTINEL,
        core_traits=[f"trait-{i}" for i in range(12)],
        cognitive_style=["系统性", "发散"],
        values=["求真", "自由"],
        motivational_drivers=["把问题想透", "被认可"],
        current_phase="职业转型期",
        life_stage="工作初期",
        deep_needs=["掌控感", "意义感"],
        preferences=PreferenceLayer(
            interests=[
                InterestTag(
                    name=f"兴趣-{i}",
                    category=f"类别-{i % 5}",
                    weight=max(0.0, 1.0 - i * 0.05),
                    first_seen="2026-01-01",
                    last_seen="2026-06-27",
                    source="behavior",
                )
                for i in range(12)
            ],
            disliked_topics=[f"不喜欢-{i}" for i in range(6)],
        ),
        recent_awareness=[
            AwarenessNote(
                date=f"2026-06-{i + 1:02d}",
                observation="近期观察" * 3,
                trend="趋势",
                emotion_guess="情绪",
            )
            for i in range(4)
        ],
        active_insights=[
            InsightHypothesis(hypothesis=f"洞察-{i}", confidence=0.7) for i in range(3)
        ],
    )


def _onion_profile() -> OnionProfile:
    """A populated ``OnionProfile`` carrying the sentinel portrait."""
    return OnionProfile(
        personality_portrait=_PORTRAIT_SENTINEL,
        core=CoreLayer(core_traits=["深究", "克制"], deep_needs=["把问题想透"]),
        values_layer=ValuesLayer(values=["求真"], motivational_drivers=["被认可"]),
        interest=InterestLayer(
            likes=[
                InterestDomain(
                    domain="国际时事",
                    weight=0.9,
                    specifics=[InterestSpecific(name="地缘政治", weight=0.8)],
                )
            ],
            dislikes=[InterestDomain(domain="八卦", weight=0.9)],
            favorite_up_users=["城市观察局"],
        ),
        role=RoleLayer(life_stage="工作初期", current_phase="职业转型期"),
        surface=SurfaceLayer(cognitive_style=["系统性"]),
    )


# --- (a) portrait-boundary sentinel exclusion --------------------------------


def test_build_profile_summary_excludes_portrait() -> None:
    profile = _soul_profile()
    assert _PORTRAIT_SENTINEL not in _canonical(build_profile_summary(profile))


def test_compact_content_prompt_summary_excludes_portrait() -> None:
    profile = _soul_profile()
    compacted = compact_content_prompt_profile_summary(build_profile_summary(profile))
    assert _PORTRAIT_SENTINEL not in _canonical(compacted)


def test_compact_gate_evaluation_summary_excludes_portrait() -> None:
    profile = _soul_profile()
    compacted = compact_gate_evaluation_profile_summary(build_profile_summary(profile))
    assert _PORTRAIT_SENTINEL not in _canonical(compacted)
    assert "recent_awareness" not in compacted


def test_query_generation_summary_excludes_portrait() -> None:
    profile = _soul_profile()
    assert _PORTRAIT_SENTINEL not in _canonical(build_query_generation_profile_summary(profile))


def test_onion_to_llm_context_excludes_portrait_when_opted_out() -> None:
    profile = _onion_profile()
    # Positive control: the portrait IS rendered under the eval/persona default…
    assert _PORTRAIT_SENTINEL in profile.to_llm_context(include_portrait=True)
    # …and MUST be absent once the content pipeline opts out.
    assert _PORTRAIT_SENTINEL not in profile.to_llm_context(include_portrait=False)


# --- (b) determinism: two calls on equal input are byte-identical ------------


def test_build_profile_summary_is_deterministic() -> None:
    p1, p2 = _soul_profile(), _soul_profile()
    assert _canonical(build_profile_summary(p1)) == _canonical(build_profile_summary(p2))


def test_compact_content_prompt_summary_is_deterministic() -> None:
    p1, p2 = _soul_profile(), _soul_profile()
    first = compact_content_prompt_profile_summary(build_profile_summary(p1))
    second = compact_content_prompt_profile_summary(build_profile_summary(p2))
    assert _canonical(first) == _canonical(second)


def test_compact_gate_evaluation_summary_is_deterministic() -> None:
    p1, p2 = _soul_profile(), _soul_profile()
    first = compact_gate_evaluation_profile_summary(build_profile_summary(p1))
    second = compact_gate_evaluation_profile_summary(build_profile_summary(p2))
    assert _canonical(first) == _canonical(second)


def test_query_generation_summary_is_deterministic() -> None:
    p1, p2 = _soul_profile(), _soul_profile()
    assert _canonical(build_query_generation_profile_summary(p1)) == _canonical(
        build_query_generation_profile_summary(p2)
    )


def test_onion_to_llm_context_is_deterministic() -> None:
    p1, p2 = _onion_profile(), _onion_profile()
    assert p1.to_llm_context(include_portrait=False) == p2.to_llm_context(include_portrait=False)


# --- speculation view (Task 7): same boundary + determinism as the fork -------


def test_speculation_view_excludes_portrait() -> None:
    """The façade ``speculation`` view keeps the portrait out on both shapes."""
    from openbiliclaw.soul import profile_views

    assert _PORTRAIT_SENTINEL not in profile_views.speculation(_onion_profile())
    assert _PORTRAIT_SENTINEL not in profile_views.speculation(_soul_profile())


def test_speculation_view_is_deterministic() -> None:
    """Two calls on equal input render byte-identically (cache-safe prompts)."""
    from openbiliclaw.soul import profile_views

    assert profile_views.speculation(_onion_profile()) == profile_views.speculation(
        _onion_profile()
    )
    assert profile_views.speculation(_soul_profile()) == profile_views.speculation(_soul_profile())
