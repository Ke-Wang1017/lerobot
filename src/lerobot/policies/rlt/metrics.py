"""RLT training metrics — three groups, three tick rates.

The collector here is structured around the principle that **distinct
event types must own distinct series with their own timestamps**, and
that **invariants are enforced at every append, not asserted in tests
after the fact**. The previous design conflated per-inference and
per-grad-update events into a single ``record_step`` method that took
optional kwargs and conditionally appended — which silently drifted
length-wise (per-inference series got 2× the entries of per-grad-update
series) and required the grad-update path to fake-pad ``actor_deltas``
with placeholder zeros. That polluted charts, broke chart x-axis
alignment, and corrupted resume.

Three event types here — three classes, three timestamp series:

* :class:`EpisodeGroup`     ticks once per ``record_episode`` call.
* :class:`InferenceGroup`   ticks once per actor inference.
* :class:`GradUpdateGroup`  ticks once per gradient-update batch.

Each group's ``append`` method takes ALL its fields by keyword (no
defaults) — adding a metric to a group is a hard error at every call
site, not a silently-skipped append. Each ``append`` re-checks the
within-group length invariant before returning. Resume goes through
``deserialize`` per group, which restores all sibling series atomically
or pads them to the shortest common length and logs.

Thread-safe: the lock lives on :class:`RLTMetrics` and wraps every
public mutation/snapshot. Group methods are NOT individually locked —
they're meant to be called from inside the aggregator's lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Group classes — each enforces its own length invariant on every mutation.
# =============================================================================


@dataclass
class EpisodeGroup:
    """One entry per ``record_episode`` call.

    Invariant: ``len(successes) == len(autonomous) == len(timestamps)
    == len(lengths_s)``.
    """

    successes: list[bool] = field(default_factory=list)
    autonomous: list[bool] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    lengths_s: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        success: bool,
        autonomous: bool,
        duration_s: float,
    ) -> None:
        self.successes.append(bool(success))
        self.autonomous.append(bool(autonomous))
        self.timestamps.append(time.time())
        self.lengths_s.append(float(duration_s))
        self._check_invariant()

    def _check_invariant(self) -> None:
        n = len(self.successes)
        assert n == len(self.autonomous) == len(self.timestamps) == len(self.lengths_s), (
            f"EpisodeGroup invariant violated: successes={n} "
            f"autonomous={len(self.autonomous)} "
            f"timestamps={len(self.timestamps)} lengths_s={len(self.lengths_s)}"
        )

    def truncate(self, max_len: int) -> None:
        if len(self.successes) > max_len:
            self.successes = self.successes[-max_len:]
            self.autonomous = self.autonomous[-max_len:]
            self.timestamps = self.timestamps[-max_len:]
            self.lengths_s = self.lengths_s[-max_len:]
        self._check_invariant()

    def __len__(self) -> int:
        return len(self.successes)

    def serialize(self) -> dict:
        return {
            "successes": list(self.successes),
            "autonomous": list(self.autonomous),
            "timestamps": list(self.timestamps),
            "lengths_s": list(self.lengths_s),
        }

    @classmethod
    def deserialize(cls, data: dict) -> EpisodeGroup:
        g = cls()
        g.successes = [bool(x) for x in data.get("successes", [])]
        g.autonomous = [bool(x) for x in data.get("autonomous", [])]
        g.timestamps = [float(x) for x in data.get("timestamps", [])]
        g.lengths_s = [float(x) for x in data.get("lengths_s", [])]
        # If sibling series have different lengths (only happens when
        # restoring from an old format that pre-dated this layout), pad
        # the shorter ones with neutral defaults so the invariant holds.
        # Loud warning so the operator knows history is partially fabricated.
        n = max(
            len(g.successes),
            len(g.autonomous),
            len(g.timestamps),
            len(g.lengths_s),
        )
        if any(len(s) != n for s in (g.successes, g.autonomous, g.timestamps, g.lengths_s)):
            logger.warning(
                "EpisodeGroup deserialize: mismatched lengths — "
                "successes=%d autonomous=%d timestamps=%d lengths_s=%d; "
                "padding shorter series to %d (front-padded with defaults).",
                len(g.successes),
                len(g.autonomous),
                len(g.timestamps),
                len(g.lengths_s),
                n,
            )
            if len(g.successes) < n:
                g.successes = [True] * (n - len(g.successes)) + g.successes
            if len(g.autonomous) < n:
                g.autonomous = [True] * (n - len(g.autonomous)) + g.autonomous
            if len(g.timestamps) < n:
                g.timestamps = [0.0] * (n - len(g.timestamps)) + g.timestamps
            if len(g.lengths_s) < n:
                g.lengths_s = [30.0] * (n - len(g.lengths_s)) + g.lengths_s
        g._check_invariant()
        return g


@dataclass
class InferenceGroup:
    """One entry per actor inference call.

    Invariant: ``len(deltas) == len(timestamps)``.
    """

    deltas: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def append(self, *, delta: float) -> None:
        self.deltas.append(float(delta))
        self.timestamps.append(time.time())
        self._check_invariant()

    def _check_invariant(self) -> None:
        n = len(self.deltas)
        assert n == len(self.timestamps), (
            f"InferenceGroup invariant violated: deltas={n} timestamps={len(self.timestamps)}"
        )

    def truncate(self, max_len: int) -> None:
        if len(self.deltas) > max_len:
            self.deltas = self.deltas[-max_len:]
            self.timestamps = self.timestamps[-max_len:]
        self._check_invariant()

    def __len__(self) -> int:
        return len(self.deltas)

    def serialize(self) -> dict:
        return {
            "deltas": list(self.deltas),
            "timestamps": list(self.timestamps),
        }

    @classmethod
    def deserialize(cls, data: dict) -> InferenceGroup:
        g = cls()
        g.deltas = [float(x) for x in data.get("deltas", [])]
        g.timestamps = [float(x) for x in data.get("timestamps", [])]
        n = max(len(g.deltas), len(g.timestamps))
        if len(g.deltas) != len(g.timestamps):
            logger.warning(
                "InferenceGroup deserialize: deltas=%d timestamps=%d; "
                "padding shorter to %d (front-padded with 0.0).",
                len(g.deltas),
                len(g.timestamps),
                n,
            )
            if len(g.deltas) < n:
                g.deltas = [0.0] * (n - len(g.deltas)) + g.deltas
            if len(g.timestamps) < n:
                g.timestamps = [0.0] * (n - len(g.timestamps)) + g.timestamps
        g._check_invariant()
        return g


@dataclass
class GradUpdateGroup:
    """One entry per gradient-update batch.

    Invariant: all series have equal length.
    """

    critic_losses: list[float] = field(default_factory=list)
    critic_grad_norms: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    q_values_mean: list[float] = field(default_factory=list)
    q_values_min: list[float] = field(default_factory=list)
    q_values_max: list[float] = field(default_factory=list)
    actor_q_terms: list[float] = field(default_factory=list)
    actor_bc_terms: list[float] = field(default_factory=list)
    update_rates: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        critic_loss: float,
        critic_grad_norm: float,
        actor_loss: float,
        q_mean: float,
        q_min: float,
        q_max: float,
        actor_q_term: float,
        actor_bc_term: float,
        update_rate: float,
    ) -> None:
        self.critic_losses.append(float(critic_loss))
        self.critic_grad_norms.append(float(critic_grad_norm))
        self.actor_losses.append(float(actor_loss))
        self.q_values_mean.append(float(q_mean))
        self.q_values_min.append(float(q_min))
        self.q_values_max.append(float(q_max))
        self.actor_q_terms.append(float(actor_q_term))
        self.actor_bc_terms.append(float(actor_bc_term))
        self.update_rates.append(float(update_rate))
        self.timestamps.append(time.time())
        self._check_invariant()

    def _all_series(self) -> tuple[list, ...]:
        return (
            self.critic_losses,
            self.critic_grad_norms,
            self.actor_losses,
            self.q_values_mean,
            self.q_values_min,
            self.q_values_max,
            self.actor_q_terms,
            self.actor_bc_terms,
            self.update_rates,
            self.timestamps,
        )

    def _check_invariant(self) -> None:
        lens = [len(s) for s in self._all_series()]
        assert len(set(lens)) == 1, (
            f"GradUpdateGroup invariant violated: "
            f"critic_losses={lens[0]} critic_grad_norms={lens[1]} "
            f"actor_losses={lens[2]} q_values_mean={lens[3]} "
            f"q_values_min={lens[4]} q_values_max={lens[5]} "
            f"actor_q_terms={lens[6]} actor_bc_terms={lens[7]} "
            f"update_rates={lens[8]} timestamps={lens[9]}"
        )

    def truncate(self, max_len: int) -> None:
        if len(self.timestamps) > max_len:
            self.critic_losses = self.critic_losses[-max_len:]
            self.critic_grad_norms = self.critic_grad_norms[-max_len:]
            self.actor_losses = self.actor_losses[-max_len:]
            self.q_values_mean = self.q_values_mean[-max_len:]
            self.q_values_min = self.q_values_min[-max_len:]
            self.q_values_max = self.q_values_max[-max_len:]
            self.actor_q_terms = self.actor_q_terms[-max_len:]
            self.actor_bc_terms = self.actor_bc_terms[-max_len:]
            self.update_rates = self.update_rates[-max_len:]
            self.timestamps = self.timestamps[-max_len:]
        self._check_invariant()

    def __len__(self) -> int:
        return len(self.timestamps)

    def serialize(self) -> dict:
        return {
            "critic_losses": list(self.critic_losses),
            "critic_grad_norms": list(self.critic_grad_norms),
            "actor_losses": list(self.actor_losses),
            "q_values_mean": list(self.q_values_mean),
            "q_values_min": list(self.q_values_min),
            "q_values_max": list(self.q_values_max),
            "actor_q_terms": list(self.actor_q_terms),
            "actor_bc_terms": list(self.actor_bc_terms),
            "update_rates": list(self.update_rates),
            "timestamps": list(self.timestamps),
        }

    @classmethod
    def deserialize(cls, data: dict) -> GradUpdateGroup:
        g = cls()
        g.critic_losses = [float(x) for x in data.get("critic_losses", [])]
        g.critic_grad_norms = [float(x) for x in data.get("critic_grad_norms", [])]
        g.actor_losses = [float(x) for x in data.get("actor_losses", [])]
        g.q_values_mean = [float(x) for x in data.get("q_values_mean", [])]
        g.q_values_min = [float(x) for x in data.get("q_values_min", [])]
        g.q_values_max = [float(x) for x in data.get("q_values_max", [])]
        g.actor_q_terms = [float(x) for x in data.get("actor_q_terms", [])]
        g.actor_bc_terms = [float(x) for x in data.get("actor_bc_terms", [])]
        g.update_rates = [float(x) for x in data.get("update_rates", [])]
        g.timestamps = [float(x) for x in data.get("timestamps", [])]
        all_series = g._all_series()
        lens = [len(s) for s in all_series]
        if len(set(lens)) != 1:
            n = min(lens)  # truncate to the shortest, drop suffixes
            logger.warning(
                "GradUpdateGroup deserialize: mismatched lengths %s; trimming all to common minimum %d.",
                lens,
                n,
            )
            g.critic_losses = g.critic_losses[:n]
            g.critic_grad_norms = g.critic_grad_norms[:n]
            g.actor_losses = g.actor_losses[:n]
            g.q_values_mean = g.q_values_mean[:n]
            g.q_values_min = g.q_values_min[:n]
            g.q_values_max = g.q_values_max[:n]
            g.actor_q_terms = g.actor_q_terms[:n]
            g.actor_bc_terms = g.actor_bc_terms[:n]
            g.update_rates = g.update_rates[:n]
            g.timestamps = g.timestamps[:n]
        g._check_invariant()
        return g


# =============================================================================
# Aggregator
# =============================================================================


@dataclass
class RLTMetrics:
    """Thread-safe metrics store for RLT training visualization.

    Owns three groups (episode / inference / grad_update) plus a few
    scalar "current value" fields. All mutations and snapshots go
    through ``self._lock``.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    episodes: EpisodeGroup = field(default_factory=EpisodeGroup)
    inferences: InferenceGroup = field(default_factory=InferenceGroup)
    grad_updates: GradUpdateGroup = field(default_factory=GradUpdateGroup)

    # Current values (scalars, overwritten on every record)
    episode: int = 0
    step_count: int = 0
    buffer_size: int = 0
    total_updates: int = 0
    mode: str = "WARMUP"

    _MAX_SERIES_LEN: int = field(default=5000, repr=False)

    # ------------ append APIs (one per event type) ------------

    def record_episode(
        self,
        episode: int,
        success: bool,
        autonomous: bool,
        duration_s: float,
    ) -> None:
        with self._lock:
            self.episode = episode
            self.episodes.append(
                success=success,
                autonomous=autonomous,
                duration_s=duration_s,
            )
            self.episodes.truncate(self._MAX_SERIES_LEN)

    def record_inference(
        self,
        step: int,
        delta: float,
        buffer_size: int,
        total_updates: int,
        mode: str,
    ) -> None:
        with self._lock:
            self.step_count = step
            self.buffer_size = buffer_size
            self.total_updates = total_updates
            self.mode = mode
            self.inferences.append(delta=delta)
            self.inferences.truncate(self._MAX_SERIES_LEN)

    def record_grad_update(
        self,
        *,
        total_updates: int,
        mode: str,
        critic_loss: float,
        critic_grad_norm: float,
        actor_loss: float,
        q_mean: float,
        q_min: float,
        q_max: float,
        actor_q_term: float,
        actor_bc_term: float,
        update_rate: float,
    ) -> None:
        with self._lock:
            self.total_updates = total_updates
            self.mode = mode
            self.grad_updates.append(
                critic_loss=critic_loss,
                critic_grad_norm=critic_grad_norm,
                actor_loss=actor_loss,
                q_mean=q_mean,
                q_min=q_min,
                q_max=q_max,
                actor_q_term=actor_q_term,
                actor_bc_term=actor_bc_term,
                update_rate=update_rate,
            )
            self.grad_updates.truncate(self._MAX_SERIES_LEN)

    # ------------ snapshot / restore ------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for the API.

        Saves RAW data (no smoothing). Smoothing happens client-side so
        repeated save→load cycles don't progressively flatten the early
        portion of the series.
        """
        with self._lock:
            successes = list(self.episodes.successes)
            autonomous = list(self.episodes.autonomous)
            timestamps = list(self.episodes.timestamps)
            list(self.episodes.lengths_s)
            n = len(successes)

            # Rolling-20 rates (kept for the dashboard's "recent" tile)
            recent_s = successes[-20:]
            success_rate = sum(recent_s) / len(recent_s) if recent_s else 0.0
            recent_pairs = list(zip(successes[-20:], autonomous[-20:], strict=False))
            auto_succ_recent = sum(1 for s, a in recent_pairs if s and a)
            autonomous_rate = auto_succ_recent / len(recent_pairs) if recent_pairs else 0.0
            recent_a = autonomous[-20:]
            intervention_rate = 1.0 - (sum(recent_a) / len(recent_a)) if recent_a else 0.0

            # Throughput: autonomous successes per 10 min of active time
            now = time.time()
            window_start = now - 600
            throughput = sum(
                1
                for s, a, t in zip(successes, autonomous, timestamps, strict=False)
                if t >= window_start and s and a
            )

            # Pre-computed rolling autonomous rate (for the chart)
            auto_rate_rolling = []
            for i in range(n):
                win_s = successes[max(0, i - 19) : i + 1]
                win_a = autonomous[max(0, i - 19) : i + 1]
                auto_s = sum(1 for s, a in zip(win_s, win_a, strict=False) if s and a)
                auto_rate_rolling.append(auto_s / len(win_s) if win_s else 0.0)

            # All-time rates
            n_total = max(n, 1)
            success_count_all = sum(successes)
            auto_succ_count_all = sum(1 for s_, a in zip(successes, autonomous, strict=False) if s_ and a)
            intervention_count_all = sum(1 for a in autonomous if not a)

            return {
                "episode": self.episode,
                "step_count": self.step_count,
                "buffer_size": self.buffer_size,
                "total_updates": self.total_updates,
                "mode": self.mode,
                # Rolling-20 (kept for the dashboard "recent" tile).
                "success_rate": round(success_rate, 3),
                "autonomous_rate": round(autonomous_rate, 3),
                "intervention_rate": round(intervention_rate, 3),
                # All-time rates over the entire run.
                "success_rate_alltime": round(success_count_all / n_total, 3),
                "autonomous_rate_alltime": round(auto_succ_count_all / n_total, 3),
                "intervention_rate_alltime": round(intervention_count_all / n_total, 3),
                "throughput_10min": throughput,
                # NOTE: ``total_autonomous`` here is the count of episodes
                # that *succeeded* AND had no intervention (the meaningful
                # number — was previously a misleading no-intervention count
                # regardless of outcome). Same metric as
                # ``autonomous_rate_alltime * total_episodes``.
                "total_successes": success_count_all,
                "total_autonomous_successes": auto_succ_count_all,
                "total_episodes": n,
                # Three groups, three independent series. Each group's
                # entries share an index; cross-group indices have no
                # alignment relationship.
                "series": {
                    "episodes": self.episodes.serialize(),
                    "inferences": self.inferences.serialize(),
                    "grad_updates": self.grad_updates.serialize(),
                    # Derived (per-episode) — already rolling-windowed.
                    "autonomous_rate_rolling": auto_rate_rolling[-200:],
                },
            }

    def restore(self, snap: dict) -> None:
        """Restore state from a snapshot dict (typically loaded via
        :func:`load_metrics_from_file`).

        Atomic per group: each group is loaded via its own
        ``deserialize`` which validates the within-group invariant.
        Logs the loaded shapes so a subsequent corruption is loud.
        """
        with self._lock:
            self.episode = int(snap.get("episode", 0))
            self.step_count = int(snap.get("step_count", 0))
            self.buffer_size = int(snap.get("buffer_size", 0))
            self.total_updates = int(snap.get("total_updates", 0))
            self.mode = str(snap.get("mode", "WARMUP"))
            series = snap.get("series", {})
            self.episodes = EpisodeGroup.deserialize(series.get("episodes", {}))
            self.inferences = InferenceGroup.deserialize(series.get("inferences", {}))
            self.grad_updates = GradUpdateGroup.deserialize(series.get("grad_updates", {}))
            logger.info(
                "RLT metrics restored: episodes=%d inferences=%d grad_updates=%d "
                "(episode=%d, total_updates=%d)",
                len(self.episodes),
                len(self.inferences),
                len(self.grad_updates),
                self.episode,
                self.total_updates,
            )


# =============================================================================
# Module-level singleton + file I/O
# =============================================================================

_global_metrics: RLTMetrics | None = None
_metrics_path: str | None = None


def get_metrics() -> RLTMetrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RLTMetrics()
    return _global_metrics


def set_metrics_path(path: str) -> None:
    global _metrics_path
    _metrics_path = path


def save_metrics_to_file() -> None:
    """Snapshot + atomic write. Logs at INFO on first success and at
    ERROR on any failure (with the exception type + message), so a stale
    file doesn't go unnoticed."""
    import json
    import os

    if _global_metrics is None or _metrics_path is None:
        return
    try:
        snap = _global_metrics.snapshot()
        tmp = _metrics_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, _metrics_path)
    except Exception as e:
        # Errors here mean the file is stale — operator needs to know.
        # ERROR not WARNING because metrics drive operator decisions.
        logger.error(
            "RLT metrics save FAILED (file is now stale): %s: %s",
            type(e).__name__,
            e,
        )


def load_metrics_from_file(path: str | None = None) -> dict:
    import json

    p = path or _metrics_path
    if not p:
        import os

        for candidate in ["outputs/rlt_online/metrics.json"]:
            if os.path.exists(candidate):
                p = candidate
                break
    if not p:
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def reset_metrics() -> None:
    global _global_metrics
    _global_metrics = RLTMetrics()
