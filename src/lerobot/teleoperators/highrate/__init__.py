"""High-rate background-poll machinery for leader teleoperators.

Any leader that wants its bus read in a background thread (so
``get_action()`` becomes a fast cached read) can opt in by:

  * inheriting :class:`HighRateLeaderMixin` alongside the base leader
    class (mixin first in MRO),
  * including :class:`HighRateLeaderConfig` fields in its config
    dataclass (typically via inheritance).

The mixin handles ``LockedBus`` wrapping, thread lifecycle, and the
``get_action`` cache-hit / cache-miss fallback.
"""

from .config import HighRateLeaderConfig
from .mixin import HighRateLeaderMixin

__all__ = ["HighRateLeaderConfig", "HighRateLeaderMixin"]
