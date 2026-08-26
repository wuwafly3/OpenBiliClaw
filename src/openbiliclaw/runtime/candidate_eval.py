"""Continuous, work-conserving discovery-candidate evaluation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openbiliclaw.llm.base import classify_llm_failure_kind

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_RATE_LIMIT_BACKOFF_SECONDS = (15.0, 30.0, 60.0, 120.0, 300.0)
_TRANSIENT_BACKOFF_SECONDS = (15.0, 30.0, 60.0, 120.0, 300.0)
_NO_PROGRESS_BACKOFF_SECONDS = (60.0, 120.0, 300.0)
# A 30s first delay is roughly 60–100 times the observed 0.3–0.5s empty refresh cost,
# preserving quick recovery from source jitter. The 600s cap guarantees self-healing
# within ten minutes after a source recovers.
_SUPPLY_UNPRODUCTIVE_BACKOFF_SECONDS = (30.0, 60.0, 120.0, 300.0, 600.0)
_MAX_CANDIDATE_EVAL_WORKERS = 3
_MAX_CANDIDATE_EVAL_BATCH_SIZE = 30


@dataclass(frozen=True)
class CandidateEvalSnapshot:
    """Durable inventory counts used for coordinator decisions."""

    available: int
    target: int
    pending_eval: int
    evaluating: int
    evaluated_pending_admission: int
    admitted_pending_copy: int
    admitted_pending_available: int | None = None
    evaluated_waiting_total: int | None = None


def effective_candidate_eval_workers(configured: int, llm_concurrency: int) -> int:
    """Reserve one global LLM slot while allocating candidate workers."""

    desired = max(1, min(_MAX_CANDIDATE_EVAL_WORKERS, int(configured)))
    global_limit = max(1, int(llm_concurrency))
    return min(desired, max(1, global_limit - 1))


def _supply_result_is_productive(result: Any) -> bool:
    """Return whether a supply request made concrete replenishment progress.

    New runtime callbacks report an explicit productivity flag or count.  The
    legacy ``refreshed`` fallback remains for third-party/one-shot callbacks,
    but an explicit zero must win over ``refreshed=True``: merely executing a
    discovery strategy does not mean it inserted a new raw candidate.
    """

    if not isinstance(result, Mapping):
        return True
    try:
        if "supply_productive" in result:
            return bool(result.get("supply_productive"))
        for key in ("supply_progress_count", "supply_inserted_count"):
            if key in result:
                return int(result.get(key, 0) or 0) > 0

        progress_keys = ("inserted", "enqueued", "cached", "discovered")
        present_progress = [key for key in progress_keys if key in result]
        if present_progress:
            return any(int(result.get(key, 0) or 0) > 0 for key in present_progress)
        return bool(result.get("refreshed"))
    except Exception:
        logger.debug(
            "candidate supply returned an unreadable result; treating it as productive",
            exc_info=True,
        )
        return True


class CandidateEvalCoordinator:
    """Own claims, parallelize LLM work, and serialize completion writes."""

    def __init__(
        self,
        *,
        pipeline: Any,
        snapshot_provider: Any,
        profile_provider: Any,
        worker_count: int = 3,
        batch_size: int = 30,
        supply_callback: Any | None = None,
        post_commit_callback: Any | None = None,
        on_admitted: Callable[[int], None] | None = None,
        work_allowed: Any | None = None,
        pre_admit_hook: Callable[[], None] | None = None,
        safety_wake_seconds: float = 60.0,
        time_fn: Any = time.monotonic,
    ) -> None:
        self.pipeline = pipeline
        self.snapshot_provider = snapshot_provider
        self.profile_provider = profile_provider
        # Pool-share fairness (spec 2026-07-20, D7): a per-tick hook run right
        # before admission — the controller wires its share rebalance + deficit
        # summary here so those Phase 3/4 behaviors are reached under the
        # production (coordinator) assembly, not only the legacy drain.
        self.pre_admit_hook = pre_admit_hook
        # The approved inventory-safe design bounds live raw work to 3×30.
        # This constructor clamp protects direct composition roots as well as
        # normal config/API validation.
        self.worker_count = max(1, min(_MAX_CANDIDATE_EVAL_WORKERS, int(worker_count)))
        self.batch_size = max(1, min(_MAX_CANDIDATE_EVAL_BATCH_SIZE, int(batch_size)))
        self.supply_callback = supply_callback
        self.post_commit_callback = post_commit_callback
        self.on_admitted = on_admitted
        self.work_allowed = work_allowed
        self.safety_wake_seconds = max(0.01, float(safety_wake_seconds))
        self.time_fn = time_fn

        self._wake_event = asyncio.Event()
        self._generation = 0
        self._workers: dict[asyncio.Task[Any], Any] = {}
        self._tick_freeze_task: asyncio.Task[tuple[Any, Any]] | None = None
        self._supply_task: asyncio.Task[Any] | None = None
        self._post_commit_task: asyncio.Task[Any] | None = None
        self._post_commit_requested = False
        self._cleanup_lock = asyncio.Lock()
        self._released_tokens: set[str] = set()
        self._stopping = False
        self._running = False
        self._paused = False
        self._backoff_until = 0.0
        self._rate_limit_streak = 0
        self._transient_streak = 0
        self._zero_cache_streak = 0
        self._no_progress_level = 0
        self._supply_streak = 0
        self._supply_cooldown_until = 0.0
        self._supply_starvation_warned = False

        self.state = "idle"
        self.last_wake_reason = ""
        self.last_error = ""
        self.last_batch_seconds = 0.0
        self.last_cached = 0
        self.last_rejected = 0

    def notify(self, reason: str) -> None:
        """Publish a level-triggered wake-up without losing boundary races."""

        self._generation += 1
        self.last_wake_reason = str(reason)
        self._supply_cooldown_until = 0.0
        resume_notification = self._resume_notification(reason)
        supply_progress_notification = str(reason).strip().lower().startswith("candidate_enqueued:")
        if resume_notification or supply_progress_notification:
            self._reset_supply_backoff()
        if self._paused and resume_notification:
            self._paused = False
            self._backoff_until = 0.0
        self._wake_event.set()

    async def run_forever(self) -> None:
        """Continuously fill open evaluator slots until stopped or at target."""

        if self._running:
            return
        self._running = True
        self.notify("startup")
        try:
            while not self._stopping:
                await self._commit_finished_workers()
                if self._stopping:
                    break

                await self._settle_supply_task()
                await self._settle_post_commit_task()
                if self.work_allowed is not None and not bool(self.work_allowed()):
                    self.state = "paused"
                    await self._wait_for_activity(self.safety_wake_seconds)
                    continue
                now = self.time_fn()
                if self._paused:
                    self.state = "paused"
                    await self._wait_for_activity(self.safety_wake_seconds)
                    continue
                if self._backoff_until > now:
                    self.state = "backoff"
                    await self._wait_for_activity(
                        min(self.safety_wake_seconds, self._backoff_until - now)
                    )
                    continue
                self._backoff_until = 0.0

                self._run_pre_admit_hook()
                snapshot = self._snapshot()
                self._admit_evaluated(snapshot)
                snapshot = self._snapshot()
                if self._projected_inventory(snapshot) >= snapshot.target:
                    self.state = "idle"
                else:
                    coalescing_wait = self._fill_open_slots()
                    snapshot = self._snapshot()
                    if not self._workers and snapshot.pending_eval <= 0:
                        supply_cooldown_remaining = self._supply_cooldown_until - now
                        if supply_cooldown_remaining > 0:
                            self.state = "supply_cooldown"
                            await self._wait_for_activity(
                                min(self.safety_wake_seconds, supply_cooldown_remaining)
                            )
                            continue
                        self._request_supply("candidate_supply")
                        self.state = "waiting_supply" if self._supply_task else "idle"
                    elif self._workers:
                        self.state = "running"
                    elif coalescing_wait is not None and coalescing_wait > 0:
                        self.state = "coalescing"
                        await self._wait_for_activity(
                            min(self.safety_wake_seconds, coalescing_wait)
                        )
                        continue

                await self._wait_for_activity(self.safety_wake_seconds)
        finally:
            self.state = "stopping"
            self._stopping = True
            await self._cleanup_workers()
            await self._cancel_supply_task()
            await self._cancel_post_commit_task()
            self._running = False

    async def stop(self) -> None:
        """Stop new claims, cancel workers, and release every unfinished token."""

        self._stopping = True
        self.state = "stopping"
        self._wake_event.set()
        await self._cleanup_workers()
        await self._cancel_supply_task()
        await self._cancel_post_commit_task()

    def status_payload(self) -> dict[str, Any]:
        """Return stable runtime diagnostics for API and event payloads."""

        snapshot = self._snapshot()
        return {
            "candidate_eval_state": self.state,
            "candidate_eval_workers": self.worker_count,
            "candidate_eval_in_flight": len(self._workers),
            "candidate_eval_pending": snapshot.pending_eval,
            "candidate_eval_backoff_until": self._backoff_until,
            "candidate_eval_supply_streak": self._supply_streak,
            "candidate_eval_supply_cooldown_until": self._supply_cooldown_until,
            "candidate_eval_last_error": self.last_error,
            "candidate_eval_last_batch_seconds": self.last_batch_seconds,
            "candidate_eval_last_cached": self.last_cached,
            "candidate_eval_last_rejected": self.last_rejected,
        }

    def _snapshot(self) -> CandidateEvalSnapshot:
        value = self.snapshot_provider()
        if isinstance(value, CandidateEvalSnapshot):
            return value
        return CandidateEvalSnapshot(
            available=int(value.get("available", 0)),
            target=int(value.get("target", 0)),
            pending_eval=int(value.get("pending_eval", 0)),
            evaluating=int(value.get("evaluating", 0)),
            evaluated_pending_admission=int(value.get("evaluated_pending_admission", 0)),
            admitted_pending_copy=int(value.get("admitted_pending_copy", 0)),
            admitted_pending_available=(
                int(value["admitted_pending_available"])
                if value.get("admitted_pending_available") is not None
                else None
            ),
            evaluated_waiting_total=(
                int(value["evaluated_waiting_total"])
                if value.get("evaluated_waiting_total") is not None
                else None
            ),
        )

    def _fill_open_slots(self) -> float | None:
        tick_freeze: asyncio.Task[tuple[Any, Any]] | None = None
        while not self._stopping and len(self._workers) < self.worker_count:
            snapshot = self._snapshot()
            if self._projected_inventory(snapshot) >= snapshot.target or snapshot.pending_eval <= 0:
                return None
            claim_ready = getattr(self.pipeline, "claim_ready_batch", None)
            if callable(claim_ready):
                claim = claim_ready(limit=self.batch_size)
            else:
                claim = self.pipeline.claim_batch(limit=self.batch_size)
            if claim is None:
                ready_in = getattr(self.pipeline, "eval_ready_in_seconds", None)
                if callable(ready_in):
                    return max(0.0, float(ready_in(limit=self.batch_size)))
                return None
            if tick_freeze is None:
                tick_freeze = self._start_tick_evaluation_freeze()
            task = asyncio.create_task(
                self._evaluate_worker(claim, tick_freeze),
                name=f"candidate_eval:{claim.token[:8]}",
            )
            self._workers[task] = claim
        return None

    def _start_tick_evaluation_freeze(self) -> asyncio.Task[tuple[Any, Any]]:
        """Load one profile + eval snapshot for every worker spawned in this fill."""

        async def load() -> tuple[Any, Any]:
            profile = self.profile_provider()
            if inspect.isawaitable(profile):
                profile = await profile
            snapshot = self._capture_tick_evaluation_snapshot(profile)
            return profile, snapshot

        freeze = asyncio.create_task(load(), name="candidate_eval:tick_snapshot")
        self._tick_freeze_task = freeze
        return freeze

    def _capture_tick_evaluation_snapshot(self, profile: Any) -> Any:
        engine = getattr(self.pipeline, "discovery_engine", None)
        capture = getattr(engine, "capture_live_evaluation_context", None)
        if not callable(capture) or profile is None:
            return None
        try:
            return capture(profile)
        except Exception:
            logger.debug("candidate eval tick snapshot capture failed", exc_info=True)
            return None

    async def _evaluate_worker(
        self,
        claim: Any,
        tick_freeze: asyncio.Task[tuple[Any, Any]] | None = None,
    ) -> Any:
        if tick_freeze is None:
            profile = self.profile_provider()
            if inspect.isawaitable(profile):
                profile = await profile
            snapshot = self._capture_tick_evaluation_snapshot(profile)
        else:
            profile, snapshot = await tick_freeze
        engine, token = self._install_tick_evaluation_snapshot(profile, snapshot)
        try:
            return await self.pipeline.evaluate_claim(claim, profile)
        finally:
            unbind = getattr(engine, "_unbind_evaluation_context", None)
            if callable(unbind):
                unbind(token)

    def _install_tick_evaluation_snapshot(self, profile: Any, snapshot: Any) -> tuple[Any, Any]:
        if snapshot is None:
            return None, None
        engine = getattr(self.pipeline, "discovery_engine", None)
        bind = getattr(engine, "_bind_evaluation_context", None)
        if not callable(bind):
            return None, None
        try:
            _frozen, token = bind(profile, snapshot=snapshot)
        except Exception:
            logger.debug("candidate eval tick snapshot bind failed", exc_info=True)
            return engine, None
        return engine, token

    def _run_pre_admit_hook(self) -> None:
        """Run the controller's per-tick share maintenance before admission.

        Pool-share fairness (spec 2026-07-20, D7). Runs before ``_admit_evaluated``
        so any slot the rebalance frees is seated by under-share supply the same
        tick. Never raises into the eval loop.
        """
        hook = self.pre_admit_hook
        if hook is None:
            return
        try:
            hook()
        except Exception:
            logger.debug("candidate eval pre-admit hook failed", exc_info=True)

    def _admit_evaluated(self, snapshot: CandidateEvalSnapshot) -> None:
        waiting_total = (
            snapshot.evaluated_pending_admission
            if snapshot.evaluated_waiting_total is None
            else max(0, snapshot.evaluated_waiting_total)
        )
        if waiting_total <= 0:
            return
        admit = getattr(self.pipeline, "admit_evaluated", None)
        if not callable(admit):
            return
        admission_headroom = max(
            0,
            snapshot.target - snapshot.available - self._eligible_pending_inventory(snapshot),
        )
        cleanup_needed = waiting_total > max(0, snapshot.evaluated_pending_admission)
        if admission_headroom <= 0 and not cleanup_needed:
            return
        result = admit(limit=admission_headroom)
        self.last_cached = int(result.get("cached", 0))
        self.last_rejected = int(result.get("rejected", 0))
        self._notify_admitted(self.last_cached)

    async def _commit_finished_workers(self) -> None:
        done = [task for task in self._workers if task.done()]
        for task in done:
            claim = self._workers.pop(task)
            try:
                outcome = task.result()
                snapshot = self._snapshot()
                admission_headroom = max(
                    0,
                    snapshot.target
                    - snapshot.available
                    - self._eligible_pending_inventory(snapshot),
                )
                result = await self.pipeline.complete_claim(
                    outcome,
                    admission_limit=admission_headroom,
                )
            except asyncio.CancelledError:
                self._release_once(claim, reason="evaluation cancelled")
                continue
            except Exception as exc:
                self._release_once(claim, reason=str(exc))
                self._record_failure(exc)
                continue

            self.last_error = ""
            self.last_batch_seconds = float(getattr(outcome, "elapsed_seconds", 0.0) or 0.0)
            self.last_cached = int(result.get("cached", 0))
            self.last_rejected = int(result.get("rejected", 0))
            self._rate_limit_streak = 0
            self._transient_streak = 0
            if int(result.get("evaluated", 0)) > 0 and self.last_cached <= 0:
                self._zero_cache_streak += 1
            elif self.last_cached > 0:
                self._zero_cache_streak = 0
                self._no_progress_level = 0
                self._reset_supply_backoff()
            if self._zero_cache_streak >= 3:
                delay = _NO_PROGRESS_BACKOFF_SECONDS[
                    min(self._no_progress_level, len(_NO_PROGRESS_BACKOFF_SECONDS) - 1)
                ]
                self._no_progress_level += 1
                self._zero_cache_streak = 0
                self._backoff_until = max(self._backoff_until, self.time_fn() + delay)
                self._request_supply("candidate_eval_no_progress")
            if self.last_cached > 0:
                self._notify_admitted(self.last_cached)
                self._request_post_commit()

    def _notify_admitted(self, cached_count: int) -> None:
        if cached_count <= 0 or self.on_admitted is None:
            return
        try:
            self.on_admitted(cached_count)
        except Exception:
            logger.warning("candidate admission callback failed", exc_info=True)

    def _record_failure(self, exc: BaseException) -> None:
        self.last_error = str(exc)
        kind = classify_llm_failure_kind(exc)
        now = self.time_fn()
        if kind == "rate_limited":
            # 15s matches the scheduler's minimum useful retry cadence. Recalibrate
            # when provider/model cooldown behavior materially changes.
            delay = _RATE_LIMIT_BACKOFF_SECONDS[
                min(self._rate_limit_streak, len(_RATE_LIMIT_BACKOFF_SECONDS) - 1)
            ]
            self._rate_limit_streak += 1
            self._backoff_until = now + max(delay, self._retry_after_seconds(exc))
            return
        if kind in {"no_provider", "auth_failed"}:
            self._paused = True
            return
        if kind not in {"timeout", "connection", "server_error"}:
            logger.warning("candidate evaluation worker failed: %s", exc)
            return
        delay = _TRANSIENT_BACKOFF_SECONDS[
            min(self._transient_streak, len(_TRANSIENT_BACKOFF_SECONDS) - 1)
        ]
        self._transient_streak += 1
        self._backoff_until = now + max(delay, self._retry_after_seconds(exc))
        logger.warning("candidate evaluation worker failed: %s", exc)

    async def _wait_for_activity(self, timeout: float) -> None:
        observed_generation = self._generation
        self._wake_event.clear()
        if observed_generation != self._generation:
            return
        wake_task = asyncio.create_task(self._wake_event.wait())
        waiters: set[asyncio.Task[Any]] = {wake_task, *self._workers.keys()}
        if self._supply_task is not None:
            waiters.add(self._supply_task)
        if self._post_commit_task is not None:
            waiters.add(self._post_commit_task)
        try:
            await asyncio.wait(
                waiters,
                timeout=max(0.0, float(timeout)),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not wake_task.done():
                wake_task.cancel()
            await asyncio.gather(wake_task, return_exceptions=True)

    def _request_supply(self, reason: str) -> None:
        if self.supply_callback is None or self._supply_task is not None:
            return
        callback = self.supply_callback

        async def run() -> Any:
            result = callback(reason)
            if inspect.isawaitable(result):
                return await result
            return result

        self._supply_task = asyncio.create_task(run(), name="candidate_eval:supply")

    async def _settle_supply_task(self) -> None:
        task = self._supply_task
        if task is None or not task.done():
            return
        self._supply_task = None
        try:
            result = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("candidate evaluation supply request failed: %s", exc)
            self.last_error = str(exc)
            self._record_supply_result(productive=False)
            return
        self._record_supply_result(productive=_supply_result_is_productive(result))

    def _record_supply_result(self, *, productive: bool) -> None:
        if productive:
            self._reset_supply_backoff()
            return
        delay = _SUPPLY_UNPRODUCTIVE_BACKOFF_SECONDS[
            min(self._supply_streak, len(_SUPPLY_UNPRODUCTIVE_BACKOFF_SECONDS) - 1)
        ]
        self._supply_streak += 1
        self._supply_cooldown_until = self.time_fn() + delay
        if delay == _SUPPLY_UNPRODUCTIVE_BACKOFF_SECONDS[-1] and not (
            self._supply_starvation_warned
        ):
            self._supply_starvation_warned = True
            logger.warning(
                "candidate supply starved: %d consecutive unproductive replenishments, "
                "cooling down %.0fs",
                self._supply_streak,
                delay,
            )

    def _reset_supply_backoff(self) -> None:
        self._supply_streak = 0
        self._supply_cooldown_until = 0.0
        self._supply_starvation_warned = False

    async def _cancel_supply_task(self) -> None:
        task = self._supply_task
        self._supply_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _request_post_commit(self) -> None:
        callback = self.post_commit_callback
        if callback is None or self._stopping:
            return
        if self._post_commit_task is not None:
            self._post_commit_requested = True
            return

        async def run() -> Any:
            result = callback()
            if inspect.isawaitable(result):
                return await result
            return result

        self._post_commit_task = asyncio.create_task(
            run(),
            name="candidate_eval:post_commit",
        )

    async def _settle_post_commit_task(self) -> None:
        task = self._post_commit_task
        if task is None or not task.done():
            return
        self._post_commit_task = None
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("candidate evaluation post-commit hook failed: %s", exc)
            self.last_error = str(exc)
        rerun = self._post_commit_requested
        self._post_commit_requested = False
        if rerun and not self._stopping:
            self._request_post_commit()

    async def _cancel_post_commit_task(self) -> None:
        task = self._post_commit_task
        self._post_commit_task = None
        self._post_commit_requested = False
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cleanup_workers(self) -> None:
        async with self._cleanup_lock:
            entries = list(self._workers.items())
            self._workers.clear()
            for task, _claim in entries:
                if not task.done():
                    task.cancel()
            if entries:
                await asyncio.gather(*(task for task, _claim in entries), return_exceptions=True)
            for _task, claim in entries:
                self._release_once(claim, reason="coordinator stopping")
            freeze = self._tick_freeze_task
            self._tick_freeze_task = None
            if freeze is not None and not freeze.done():
                freeze.cancel()
                await asyncio.gather(freeze, return_exceptions=True)

    def _release_once(self, claim: Any, *, reason: str) -> None:
        token = str(getattr(claim, "token", ""))
        if token in self._released_tokens:
            return
        self._released_tokens.add(token)
        self.pipeline.release_claim(claim, reason=reason, increment_attempts=False)

    @staticmethod
    def _projected_inventory(snapshot: CandidateEvalSnapshot) -> int:
        return (
            max(0, snapshot.available)
            + CandidateEvalCoordinator._eligible_pending_inventory(snapshot)
            + max(0, snapshot.evaluated_pending_admission)
        )

    @staticmethod
    def _eligible_pending_inventory(snapshot: CandidateEvalSnapshot) -> int:
        """Return pending-copy rows that can fill a public inventory slot.

        ``None`` preserves compatibility with older snapshot providers while
        production composition roots always inject the canonical eligible
        count from storage.
        """

        value = snapshot.admitted_pending_available
        if value is None:
            value = snapshot.admitted_pending_copy
        return max(0, int(value))

    @staticmethod
    def _resume_notification(reason: str) -> bool:
        normalized = str(reason).strip().lower()
        return normalized == "startup" or normalized.startswith(("config_", "manual_"))

    @staticmethod
    def _retry_after_seconds(exc: BaseException) -> float:
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            value = getattr(current, "retry_after", None)
            if isinstance(value, int | float) and value > 0:
                return float(value)
            current = current.__cause__ or current.__context__
        return 0.0
