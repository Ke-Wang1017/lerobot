"""Tests for the RLT metrics collector.

The previous design conflated per-inference and per-grad-update events
into one method, with optional kwargs and conditional appends. This led
to silent length divergence across "sibling" series and a polluted
``actor_deltas`` chart. The refactor groups events into three classes
(``EpisodeGroup``, ``InferenceGroup``, ``GradUpdateGroup``), each
enforcing its own length invariant on every append.

These tests lock in the invariants so that future refactors that break
group atomicity fail loudly instead of silently corrupting charts."""

from __future__ import annotations

import pytest

from lerobot.policies.hvla.rlt.metrics import (
    EpisodeGroup,
    GradUpdateGroup,
    InferenceGroup,
    RLTMetrics,
    get_metrics,
    reset_metrics,
)

# ============================================================================
# Per-group invariants — within-group length parity is the contract
# ============================================================================


class TestEpisodeGroupInvariant:
    def test_append_grows_all_series_in_lockstep(self):
        g = EpisodeGroup()
        for i in range(5):
            g.append(success=True, autonomous=False, duration_s=10.0)
            assert len(g.successes) == len(g.autonomous) == len(g.timestamps) == len(g.lengths_s) == i + 1

    def test_truncate_preserves_invariant(self):
        g = EpisodeGroup()
        for _ in range(50):
            g.append(success=True, autonomous=True, duration_s=1.0)
        g.truncate(20)
        assert len(g.successes) == len(g.autonomous) == len(g.timestamps) == len(g.lengths_s) == 20

    def test_invariant_check_catches_external_corruption(self):
        """If something bypasses ``append`` and pokes a single series,
        ``_check_invariant`` must detect the drift."""
        g = EpisodeGroup()
        g.append(success=True, autonomous=True, duration_s=1.0)
        g.successes.append(False)  # corrupt only this series
        with pytest.raises(AssertionError, match="EpisodeGroup invariant"):
            g._check_invariant()


class TestInferenceGroupInvariant:
    def test_append_keeps_deltas_and_timestamps_aligned(self):
        g = InferenceGroup()
        for i in range(10):
            g.append(delta=0.05)
            assert len(g.deltas) == len(g.timestamps) == i + 1

    def test_truncate_preserves_invariant(self):
        g = InferenceGroup()
        for _ in range(100):
            g.append(delta=0.01)
        g.truncate(30)
        assert len(g.deltas) == len(g.timestamps) == 30

    def test_invariant_catches_drift(self):
        g = InferenceGroup()
        g.append(delta=0.0)
        g.deltas.append(99.0)
        with pytest.raises(AssertionError, match="InferenceGroup invariant"):
            g._check_invariant()


class TestGradUpdateGroupInvariant:
    def _payload(self):
        return {
            "critic_loss": 0.01,
            "critic_grad_norm": 2.0,
            "actor_loss": -0.3,
            "q_mean": 0.5,
            "q_min": -0.5,
            "q_max": 1.0,
            "actor_q_term": -0.3,
            "actor_bc_term": 0.1,
            "update_rate": 10.0,
        }

    def test_all_ten_series_grow_together(self):
        g = GradUpdateGroup()
        for i in range(7):
            g.append(**self._payload())
            n = i + 1
            assert all(len(s) == n for s in g._all_series()), (
                "all 10 series must have the same length after every append"
            )

    def test_truncate_preserves_invariant(self):
        g = GradUpdateGroup()
        for _ in range(100):
            g.append(**self._payload())
        g.truncate(25)
        assert all(len(s) == 25 for s in g._all_series())

    def test_invariant_catches_any_single_series_drift(self):
        """All ten series form one group. Any one going out of sync must
        trip the assert — covers regressions where a future contributor
        adds a new field but forgets to update one append site."""
        g = GradUpdateGroup()
        g.append(**self._payload())
        g.q_values_max.append(2.0)  # just one drifts
        with pytest.raises(AssertionError, match="GradUpdateGroup invariant"):
            g._check_invariant()

    def test_append_requires_all_kwargs(self):
        """Adding a new metric to the group should be a hard error at
        every call site. Verify by trying to call append with one
        kwarg missing."""
        g = GradUpdateGroup()
        partial = self._payload()
        del partial["q_max"]
        with pytest.raises(TypeError):
            g.append(**partial)


# ============================================================================
# Round-trip serialize/deserialize — resume must preserve every field
# ============================================================================


class TestRoundTrip:
    def _drive_metrics(self) -> RLTMetrics:
        m = RLTMetrics()
        for ep in range(3):
            m.record_episode(
                episode=ep,
                success=(ep != 1),
                autonomous=(ep == 0),
                duration_s=10.0 + ep,
            )
        for i in range(7):
            m.record_inference(
                step=i,
                delta=0.01 * (i + 1),
                buffer_size=100 + i,
                total_updates=10 * i,
                mode="RL",
            )
        for i in range(5):
            m.record_grad_update(
                total_updates=20 + i,
                mode="RL",
                critic_loss=0.001 * (i + 1),
                critic_grad_norm=2.0 + i,
                actor_loss=-0.3 - 0.01 * i,
                q_mean=0.5 + 0.1 * i,
                q_min=-0.4 - 0.05 * i,
                q_max=0.9 + 0.1 * i,
                actor_q_term=-0.3 - 0.01 * i,
                actor_bc_term=0.1 + 0.005 * i,
                update_rate=12.0 + i,
            )
        return m

    def test_snapshot_then_restore_preserves_all_groups(self):
        m1 = self._drive_metrics()
        snap = m1.snapshot()

        m2 = RLTMetrics()
        m2.restore(snap)

        # All three groups must come back with the same lengths
        assert len(m2.episodes) == len(m1.episodes)
        assert len(m2.inferences) == len(m1.inferences)
        assert len(m2.grad_updates) == len(m1.grad_updates)

        # Spot-check that a few values made the round trip
        assert m2.episodes.successes == m1.episodes.successes
        assert m2.inferences.deltas == m1.inferences.deltas
        assert m2.grad_updates.q_values_max == m1.grad_updates.q_values_max

        # Scalars preserved
        assert m2.episode == m1.episode
        assert m2.total_updates == m1.total_updates

    def test_restore_into_existing_metrics_replaces_groups(self):
        """A fresh ``RLTMetrics`` is born with empty groups. Restoring
        from a non-empty snapshot must completely replace them — not
        append to them — otherwise resume would double the data."""
        m1 = self._drive_metrics()
        snap = m1.snapshot()

        m2 = RLTMetrics()
        # Simulate the daemon already having recorded a few entries
        # in the new run BEFORE restore is called.
        m2.record_inference(step=0, delta=999.0, buffer_size=0, total_updates=0, mode="X")
        assert len(m2.inferences) == 1

        m2.restore(snap)
        # After restore, the in-memory data is the snapshot's data,
        # not the snapshot's data plus the local pre-restore append.
        assert len(m2.inferences) == len(m1.inferences)
        assert 999.0 not in m2.inferences.deltas

    def test_legacy_episode_group_partial_restore_pads(self):
        """If a snapshot has only some of the episode-group sibling series
        (because it was saved by an older format), deserialize pads the
        shorter siblings rather than raising. Logs a warning."""
        snap_partial_episodes = {
            "successes": [True, False, True],
            # "autonomous" missing
            # "timestamps" missing
            # "lengths_s" partial
            "lengths_s": [10.0],
        }
        g = EpisodeGroup.deserialize(snap_partial_episodes)
        assert len(g) == 3
        # Padded values should be defaults
        assert len(g.autonomous) == 3
        assert len(g.timestamps) == 3
        assert len(g.lengths_s) == 3
        # Successes preserved
        assert g.successes == [True, False, True]


# ============================================================================
# Aggregator-level — record methods preserve all invariants
# ============================================================================


class TestRLTMetrics:
    def setup_method(self):
        reset_metrics()

    def test_groups_grow_independently(self):
        """Per-inference and per-grad-update have different tick rates;
        recording one must not affect the other."""
        m = get_metrics()
        for _ in range(5):
            m.record_inference(step=0, delta=0.05, buffer_size=0, total_updates=0, mode="RL")
        assert len(m.inferences) == 5
        assert len(m.grad_updates) == 0  # no piggyback

        for _ in range(3):
            m.record_grad_update(
                total_updates=0,
                mode="RL",
                critic_loss=0.01,
                critic_grad_norm=1.0,
                actor_loss=0.0,
                q_mean=0.5,
                q_min=-0.5,
                q_max=1.0,
                actor_q_term=0.0,
                actor_bc_term=0.0,
                update_rate=10.0,
            )
        assert len(m.inferences) == 5  # unchanged
        assert len(m.grad_updates) == 3

    def test_record_inference_does_not_pollute_grad_metrics(self):
        """Locked-in: the prior bug was a fake delta=0 written from grad
        update path into actor_deltas. Verify grad-update doesn't append
        to inference.deltas, and vice versa."""
        m = get_metrics()
        m.record_inference(step=0, delta=0.123, buffer_size=0, total_updates=0, mode="RL")
        m.record_grad_update(
            total_updates=1,
            mode="RL",
            critic_loss=0.01,
            critic_grad_norm=1.0,
            actor_loss=0.0,
            q_mean=0.5,
            q_min=-0.5,
            q_max=1.0,
            actor_q_term=0.0,
            actor_bc_term=0.0,
            update_rate=10.0,
        )
        # The inference series got exactly one entry, and it's the real one
        assert m.inferences.deltas == [0.123]
        # The grad-update series got exactly one entry, with no fake delta
        assert len(m.grad_updates) == 1
        assert m.grad_updates.q_values_max == [1.0]
