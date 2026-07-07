"""Predictive-lookahead controller machinery — reusable across robots.

Any robot whose ``self.bus`` is a ``LockedBus``-wrapped ``SerialMotorsBus``
can opt into the predictive controller by:

  * inheriting :class:`PredictiveLookaheadMixin` alongside the base robot
    class (mixin first in MRO),
  * including :class:`PredictiveControllerConfig` fields in its config
    dataclass (typically via inheritance).

The mixin handles ``LockedBus`` wrapping, controller-thread lifecycle,
``send_action``/``get_observation`` overrides, and ``attach_teleop``.
The controller logic itself is motor-count-agnostic — it reads
``self.bus.motors`` at construction.
"""

from .config import PredictiveControllerConfig
from .controller import PredictiveLookaheadController
from .mixin import PredictiveLookaheadMixin

__all__ = [
    "PredictiveControllerConfig",
    "PredictiveLookaheadController",
    "PredictiveLookaheadMixin",
]
