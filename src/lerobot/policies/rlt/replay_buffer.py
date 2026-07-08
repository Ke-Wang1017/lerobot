"""Chunk-level replay buffer for RLT online RL.

Stores (z_rl, state, action_chunk, ref_chunk, reward, next_z_rl, next_state,
next_ref_chunk, done) tuples. z_rl is pre-computed and stored so replay
doesn't require re-running the frozen S1 encoder or RL token encoder.
"""

from __future__ import annotations

import threading

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity ring buffer for chunk-level transitions.

    Thread-safe: all public methods are guarded by a lock.
    Inference thread writes during RL mode, main thread writes during
    intervention — concurrent access is possible.
    """

    def __init__(
        self,
        capacity: int,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        device: torch.device,
    ):
        self._lock = threading.Lock()
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0

        A = chunk_length * action_dim
        # Store everything on CPU, move to device on sample
        self._z_rl = torch.zeros(capacity, rl_token_dim)
        self._state = torch.zeros(capacity, state_dim)
        self._action = torch.zeros(capacity, A)
        self._ref = torch.zeros(capacity, A)
        self._reward = torch.zeros(capacity, 1)
        self._next_z_rl = torch.zeros(capacity, rl_token_dim)
        self._next_state = torch.zeros(capacity, state_dim)
        self._next_ref = torch.zeros(capacity, A)
        self._done = torch.zeros(capacity, 1)

    def add(
        self,
        z_rl: Tensor,
        state: Tensor,
        action_chunk: Tensor,
        ref_chunk: Tensor,
        reward: float,
        next_z_rl: Tensor,
        next_state: Tensor,
        next_ref_chunk: Tensor,
        done: bool,
    ) -> None:
        """Add a single transition (all tensors should be 1D/2D, no batch dim)."""
        with self._lock:
            i = self.ptr
            self._z_rl[i] = z_rl.detach().cpu()
            self._state[i] = state.detach().cpu()
            self._action[i] = action_chunk.detach().cpu().reshape(-1)
            self._ref[i] = ref_chunk.detach().cpu().reshape(-1)
            self._reward[i, 0] = reward
            self._next_z_rl[i] = next_z_rl.detach().cpu()
            self._next_state[i] = next_state.detach().cpu()
            self._next_ref[i] = next_ref_chunk.detach().cpu().reshape(-1)
            self._done[i, 0] = float(done)

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def truncate(self, target_size: int) -> int:
        """Roll back the buffer to ``target_size`` entries.

        Used by the operator's "ignore current episode" key: any
        transitions added during the current episode are dropped so the
        episode is effectively never recorded. Returns the number of
        entries that were discarded (for the log line).

        Precondition: ``0 <= target_size <= self.size``. Calling with a
        target larger than the current size is treated as a programmer
        error — the asserts trip rather than silently no-op or corrupt
        the buffer.

        Limitation: only safe when no eviction has occurred since
        ``target_size`` was captured (i.e. the buffer has not wrapped
        around its ring). With our 200K capacity and runs in the
        thousands, that condition holds in practice. The asserts catch
        the case where it doesn't.
        """
        with self._lock:
            assert 0 <= target_size <= self.size, (
                f"truncate target_size={target_size} out of bounds [0, {self.size}]"
            )
            assert self.size < self.capacity or target_size == self.size, (
                "ReplayBuffer.truncate is unsafe once the ring buffer "
                "has wrapped around capacity. Refusing to truncate a "
                "filled-and-evicting buffer."
            )
            dropped = self.size - target_size
            self.size = target_size
            # ptr should track size when no wrap has occurred (which is
            # required by the assert above).
            self.ptr = target_size
            return dropped

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """Sample a random batch and move to device."""
        with self._lock:
            idx = torch.randint(0, self.size, (batch_size,))
            return {
                "z_rl": self._z_rl[idx].to(self.device),
                "state": self._state[idx].to(self.device),
                "action": self._action[idx].to(self.device),
                "ref": self._ref[idx].to(self.device),
                "reward": self._reward[idx].to(self.device),
                "next_z_rl": self._next_z_rl[idx].to(self.device),
                "next_state": self._next_state[idx].to(self.device),
                "next_ref": self._next_ref[idx].to(self.device),
                "done": self._done[idx].to(self.device),
            }

    def save(self, path: str) -> None:
        """Save buffer contents to disk. Atomic via tmp+rename so a
        crash mid-write can't leave a torn file at ``path``."""
        import os

        with self._lock:
            tmp = path + ".tmp"
            torch.save(
                {
                    "size": self.size,
                    "ptr": self.ptr,
                    "capacity": self.capacity,
                    "z_rl": self._z_rl[: self.capacity],
                    "state": self._state[: self.capacity],
                    "action": self._action[: self.capacity],
                    "ref": self._ref[: self.capacity],
                    "reward": self._reward[: self.capacity],
                    "next_z_rl": self._next_z_rl[: self.capacity],
                    "next_state": self._next_state[: self.capacity],
                    "next_ref": self._next_ref[: self.capacity],
                    "done": self._done[: self.capacity],
                },
                tmp,
            )
            os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Load buffer contents from disk. Supports same or larger capacity."""
        import logging

        with self._lock:
            data = torch.load(path, weights_only=True)
            saved_cap = data["capacity"]
            saved_size = data["size"]

            if saved_cap > self.capacity:
                raise ValueError(
                    f"Cannot load buffer: saved capacity {saved_cap} > current {self.capacity}. "
                    f"Increase replay_capacity to at least {saved_cap}."
                )

            n = saved_size
            self._z_rl[:n] = data["z_rl"][:n]
            self._state[:n] = data["state"][:n]
            self._action[:n] = data["action"][:n]
            self._ref[:n] = data["ref"][:n]
            self._reward[:n] = data["reward"][:n]
            self._next_z_rl[:n] = data["next_z_rl"][:n]
            self._next_state[:n] = data["next_state"][:n]
            self._next_ref[:n] = data["next_ref"][:n]
            self._done[:n] = data["done"][:n]
            self.size = n
            self.ptr = n % self.capacity

            if saved_cap < self.capacity:
                logging.getLogger(__name__).info(
                    "Replay buffer expanded: %d → %d capacity, %d transitions preserved",
                    saved_cap,
                    self.capacity,
                    n,
                )

    def __len__(self) -> int:
        with self._lock:
            return self.size


class TransactionalReplayBuffer:
    """Replay buffer with database-style transaction semantics:
    ``add()`` writes are *pending* until ``commit()`` (= move into the
    underlying committed buffer) or ``discard()`` (= drop them). The
    pending writes are invisible to ``sample()`` and to ``len()`` —
    the only externally observable state is the committed buffer.

    Used here at episode granularity: each episode is one
    transaction. ``commit()`` runs at success / abort / fail / timeout
    (every outcome that should keep the data); ``discard()`` runs at
    IGNORE (the operator pressed DOWN to throw out the episode).

    **Invariant**: ``sample()`` returns ONLY committed transitions.
    Pending writes are invisible to the gradient-update sampling
    path until the transaction ends. This makes IGNORE provably
    zero-impact on training: discarded writes never reach the
    sampler, so the critic and actor never see them.

    The previous design (``ReplayBuffer.add`` directly +
    ``truncate(ep_start_size)`` on IGNORE) had a structural leak:
    grad updates fired during the episode could sample the
    in-progress transitions before the truncate call ever ran. Small
    in expectation but non-zero.

    Drop-in replacement at the public API level (``add`` / ``sample``
    / ``__len__`` / ``capacity`` / ``save`` / ``load``). Private
    attribute access (``_action``, ``_z_rl``, ``ptr``, etc.) forwards
    to the committed buffer via ``__getattr__`` so existing call sites
    that poke internals (e.g. for cross-module assertions) keep
    working.

    Cost: writes from transaction N only become sampleable after
    ``commit()`` lands them in committed (i.e., starting in
    transaction N+1 if commits are scoped to transactions). With
    UTD=5 and ~25 grad updates/s wall-clock, that's ~250 grad steps
    of staleness per transaction boundary. Acceptable trade for the
    structural guarantee.
    """

    def __init__(
        self,
        capacity: int,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        device: torch.device,
    ):
        self._lock = threading.Lock()
        self._committed = ReplayBuffer(
            capacity,
            rl_token_dim,
            state_dim,
            action_dim,
            chunk_length,
            device,
        )
        # Pending writes are a list of dicts (the kwargs that ``add``
        # was called with). One per transition. At ~6 Hz inference and
        # ~30s episodes, max ~180 entries per transaction — small enough
        # that a Python list is fine. No pre-allocation pressure.
        self._pending: list[dict] = []

    # ---------------------------------------------------------------------
    # Inference-thread / recorder-thread API
    # ---------------------------------------------------------------------

    def add(
        self,
        z_rl: Tensor,
        state: Tensor,
        action_chunk: Tensor,
        ref_chunk: Tensor,
        reward: float,
        next_z_rl: Tensor,
        next_state: Tensor,
        next_ref_chunk: Tensor,
        done: bool,
    ) -> None:
        """Append a pending transition. Does NOT touch the committed
        buffer; ``sample()`` won't see this until ``commit()``."""
        # Detach + cpu now so re-using the same tensor for next chunk
        # doesn't retroactively mutate this entry. Same contract as
        # ``ReplayBuffer.add``.
        item = {
            "z_rl": z_rl.detach().cpu(),
            "state": state.detach().cpu(),
            "action_chunk": action_chunk.detach().cpu(),
            "ref_chunk": ref_chunk.detach().cpu(),
            "reward": float(reward),
            "next_z_rl": next_z_rl.detach().cpu(),
            "next_state": next_state.detach().cpu(),
            "next_ref_chunk": next_ref_chunk.detach().cpu(),
            "done": bool(done),
        }
        with self._lock:
            self._pending.append(item)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """Sample from the committed buffer ONLY. Pending writes are
        invisible to the grad-update path — this is the structural
        guarantee."""
        return self._committed.sample(batch_size)

    # ---------------------------------------------------------------------
    # Transaction boundary (called from main thread)
    # ---------------------------------------------------------------------

    def commit(self) -> int:
        """Move all pending transitions into the committed buffer.
        Called at the end of a transaction (success / abort / fail /
        timeout — every outcome that should keep the data).

        Returns the number of transitions committed. Concurrent
        ``sample()`` calls during the loop see partial commit
        states but each individual sample is internally consistent
        (committed buffer's per-call lock keeps each ``add`` atomic)."""
        with self._lock:
            n = len(self._pending)
            for item in self._pending:
                self._committed.add(**item)
            self._pending = []
        return n

    def discard(self) -> int:
        """Drop all pending writes without committing. Called at the
        end of a transaction that should be undone (IGNORE) — the
        discarded transitions never enter the committed buffer and
        so are never sampled.

        Returns the number of transitions discarded (for the log)."""
        with self._lock:
            n = len(self._pending)
            self._pending = []
        return n

    @property
    def pending_size(self) -> int:
        with self._lock:
            return len(self._pending)

    def peek_last_pending(self) -> dict | None:
        """Return the most recently added pending transition's kwargs
        (or None if there are no pending writes). Used by cross-module
        assertions that need to verify the recorder wrote what it
        claimed (e.g. ``_rlt_flush_intervention_terminal``)."""
        with self._lock:
            return self._pending[-1] if self._pending else None

    # ---------------------------------------------------------------------
    # Persistence — only the committed buffer is saved.
    # Staging is per-run, per-episode, volatile by design.
    # ---------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save committed contents. Pending writes are intentionally
        NOT persisted — they're volatile per-transaction state. If a
        crash happens mid-transaction, the next run starts with no
        pending writes and the in-progress transaction is effectively
        lost (which is correct: we don't know its outcome yet)."""
        self._committed.save(path)

    def load(self, path: str) -> None:
        self._committed.load(path)

    def __len__(self) -> int:
        """Length = committed size. Pending writes are invisible at
        this level too, matching the ``sample`` invariant."""
        return len(self._committed)

    # ---------------------------------------------------------------------
    # Forward private/public attribute access to committed for code
    # that pokes internals (audit scripts, tests, cross-module asserts).
    # ---------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Only fires on attributes not found through normal lookup
        # (i.e. anything not set on self). Forward to committed.
        # Guard against recursion if _committed isn't set yet (during
        # __init__ before assignment, or in pathological cases).
        try:
            committed = object.__getattribute__(self, "_committed")
        except AttributeError as e:
            raise AttributeError(name) from e
        return getattr(committed, name)
