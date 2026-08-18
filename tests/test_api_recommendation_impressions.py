"""Four-surface impression logging tests for the ML ranking work (Wave 0).

The exposure ledger is only usable as training data if every surface a user can
actually see recommendations on writes to it, and if a polling client cannot
flood it. These tests drive the real FastAPI app so the surface classification
and debounce live under test rather than in a docstring.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from openbiliclaw.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_runtime_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Mirror ``test_api_app.py``'s isolation so create_app() leaves degraded mode.

    ``create_app`` loads runtime config up front and returns 503 on every route
    when no LLM provider resolves. Without this the suite would pass or fail
    depending on whether the developer machine has a private config.toml.
    """
    from openbiliclaw.api import app as api_app
    from openbiliclaw.config import Config, save_config
    from openbiliclaw.runtime import embedding_progress

    embedding_progress.reset()
    project_root = tmp_path / "runtime"
    monkeypatch.setenv("OPENBILICLAW_PROJECT_ROOT", str(project_root))
    cfg = Config()
    cfg.llm.default_provider = "ollama"
    cfg.llm.ollama.model = "llama3"
    save_config(cfg, project_root / "config.toml")
    yield

    deadline = time.monotonic() + 0.5
    while any(not task.done() for task in api_app._fire_and_forget_tasks):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    for task in tuple(api_app._fire_and_forget_tasks):
        if not task.done():
            task.cancel()
    embedding_progress.reset()


_ROW: dict[str, Any] = {
    "id": 11,
    "bvid": "BV1IMPRESSION",
    "item_key": "bilibili:BV1IMPRESSION",
    "title": "讲透排序模型",
    "up_name": "算法观察局",
    "cover_url": "https://i0.hdslb.com/bfs/archive/cover.jpg",
    "expression": "这条应该正好在你最近的兴趣线上。",
    "topic": "你最近在想的排序问题",
    "confidence": 0.83,
    "presented": 0,
    "franchise_key": "",
    "source_platform": "bilibili",
    "duration": 600,
    "view_count": 1000,
}


class _FakeDatabase:
    """Minimal recommendation-serving database that records impression writes."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else [dict(_ROW)]
        self.impressions: list[list[dict[str, Any]]] = []
        self.presented_calls: list[list[int]] = []

    def get_recommendations(
        self,
        limit: int = 20,
        *,
        exclude_processed: bool = False,
    ) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows]

    def record_recommendation_impressions(
        self,
        impressions: list[dict[str, Any]],
    ) -> int:
        self.impressions.append([dict(entry) for entry in impressions])
        return len(impressions)

    def mark_recommendations_presented(self, recommendation_ids: list[int]) -> None:
        self.presented_calls.append(list(recommendation_ids))


def _flat(database: _FakeDatabase) -> list[dict[str, Any]]:
    return [entry for batch in database.impressions for entry in batch]


class TestImpressionSurfaceClassification:
    def test_extension_origin_is_logged_as_extension(self) -> None:
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        response = client.get(
            "/api/recommendations",
            headers={"origin": "chrome-extension://abcdefghijklmnop"},
        )

        assert response.status_code == 200
        entries = _flat(database)
        assert len(entries) == 1
        assert entries[0]["surface"] == "extension"
        assert entries[0]["recommendation_id"] == 11
        assert entries[0]["position"] == 0

    def test_desktop_web_referer_is_logged_as_desktop_web(self) -> None:
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        client.get(
            "/api/recommendations",
            headers={"referer": "http://127.0.0.1:8420/web/index.html"},
        )

        assert [entry["surface"] for entry in _flat(database)] == ["desktop_web"]

    def test_mobile_web_referer_is_logged_as_mobile_web(self) -> None:
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        client.get(
            "/api/recommendations",
            headers={"referer": "http://127.0.0.1:8420/m/"},
        )

        assert [entry["surface"] for entry in _flat(database)] == ["mobile_web"]

    def test_unclassifiable_request_is_logged_as_unknown(self) -> None:
        """A mislabelled exposure still trains; a dropped one is a lost negative."""
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        client.get("/api/recommendations")

        assert [entry["surface"] for entry in _flat(database)] == ["unknown"]


class TestImpressionLoggingBehaviour:
    def test_position_reflects_window_order(self) -> None:
        rows = [
            {**_ROW, "id": 1, "bvid": "BV1", "item_key": "bilibili:BV1"},
            {**_ROW, "id": 2, "bvid": "BV2", "item_key": "bilibili:BV2"},
            {**_ROW, "id": 3, "bvid": "BV3", "item_key": "bilibili:BV3"},
        ]
        database = _FakeDatabase(rows)
        client = TestClient(create_app(database=database))

        client.get("/api/recommendations")

        entries = _flat(database)
        assert [(entry["recommendation_id"], entry["position"]) for entry in entries] == [
            (1, 0),
            (2, 1),
            (3, 2),
        ]

    def test_repeated_polling_of_an_unchanged_window_is_debounced(self) -> None:
        """The endpoint is polled continuously; it must not become a write loop."""
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        for _ in range(5):
            client.get("/api/recommendations")

        assert len(database.impressions) == 1

    def test_cache_hit_still_logs_for_a_different_surface(self) -> None:
        """Two surfaces showing the same cached window are two real exposures."""
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        client.get(
            "/api/recommendations",
            headers={"origin": "chrome-extension://abcdefghijklmnop"},
        )
        client.get(
            "/api/recommendations",
            headers={"referer": "http://127.0.0.1:8420/m/"},
        )

        assert {entry["surface"] for entry in _flat(database)} == {"extension", "mobile_web"}

    def test_logging_never_marks_rows_presented(self) -> None:
        """``presented`` owns the unread badge and notification gate — not us."""
        database = _FakeDatabase()
        client = TestClient(create_app(database=database))

        client.get("/api/recommendations")

        assert database.presented_calls == []

    def test_storage_failure_does_not_break_the_response(self) -> None:
        database = _FakeDatabase()

        def _boom(impressions: list[dict[str, Any]]) -> int:
            raise RuntimeError("ledger unavailable")

        database.record_recommendation_impressions = _boom  # type: ignore[method-assign]
        client = TestClient(create_app(database=database))

        response = client.get("/api/recommendations")

        assert response.status_code == 200
        assert len(response.json()["items"]) == 1

    def test_storage_failure_does_not_debounce_retries(self) -> None:
        """A failed ledger write must not lock the same window out for 60 seconds."""
        database = _FakeDatabase()
        calls = {"n": 0}
        original = database.record_recommendation_impressions

        def _flaky(impressions: list[dict[str, Any]]) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("ledger unavailable")
            return original(impressions)

        database.record_recommendation_impressions = _flaky  # type: ignore[method-assign]
        client = TestClient(create_app(database=database))

        first = client.get("/api/recommendations")
        second = client.get("/api/recommendations")

        assert first.status_code == 200
        assert second.status_code == 200
        assert calls["n"] == 2
        assert len(database.impressions) == 1

    def test_legacy_database_without_the_ledger_is_tolerated(self) -> None:
        class LegacyDatabase:
            def get_recommendations(
                self,
                limit: int = 20,
                *,
                exclude_processed: bool = False,
            ) -> list[dict[str, Any]]:
                return [dict(_ROW)]

        client = TestClient(create_app(database=LegacyDatabase()))

        response = client.get("/api/recommendations")

        assert response.status_code == 200
        assert len(response.json()["items"]) == 1
