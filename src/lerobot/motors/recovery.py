# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Non-physical recovery of wedged motor chains.

Use when the normal robot connect path fails because one or more motors are
silent on PING (overload latch, hung MCU) and the resulting torque-on state
on the surviving motors makes the arm unusable. The recovery flow:

    1. Open each bus directly, bypassing the strict handshake.
    2. Ping every expected motor with retry.
    3. Disable torque on responders so the arm is back-drivable.
    4. Apply a driver-specific trick to non-responders:
         - Feetech: write ``Torque_Enable=0`` directly to clear an overload
           latch, then re-ping. Many overload trips clear without a power
           cycle this way.
         - Dynamixel: would call ``INST_REBOOT``. Not implemented yet.
    5. Disconnect.

Anything still unreachable after that is a hardware issue (truly hung MCU,
loose connector, dead motor) and needs physical intervention.

This is deliberately a free-function utility module rather than methods on
``MotorsBus`` / ``Robot`` so that adding recovery does not enlarge the
public surface of those classes. Driver-specific behaviour is selected by
``isinstance`` checks here rather than overrides — fine for two or three
drivers; if it grows, refactor to a dispatch registry.

Validation
----------
This module is covered by two complementary layers:

* **Unit tests** (``tests/motors/test_recover.py``) patch the bus's
  ``_connect`` / ``ping`` / ``disable_torque`` / ``_write`` methods with
  ``unittest.mock`` and exercise the control-flow on a real
  ``FeetechMotorsBus`` instance. They verify branching, retry ordering,
  report accumulation, address lookup, robot-level introspection (single
  arm, bi-arm via nested ``Robot``, no-bus placeholder), and JSON
  serialisation. They do **not** exercise the wire protocol or motor
  firmware behaviour.

* **Hardware reproduction** (one-shot experiment, not a CI test). The
  STS3215 overload-latch path was validated end-to-end on real motors by
  inducing the trapped state (lowered ``Protection_Current`` /
  ``Protection_Time`` + a hand-blocked continuous-rotation joint), running
  ``recover_bus``, observing ``status=0x20`` (overload bit) clear on the
  follow-up ``Torque_Enable=0`` write, and confirming the motor returns to
  the bus without a power cycle. This validates the wire protocol contract
  the unit tests assume but cannot exercise. To re-run it on a different
  rig or after a firmware update, the procedure is: pick a continuous
  joint (no mechanical end-stop), save its ``Protection_Current`` and
  ``Protection_Time``, lower them aggressively, command a goal, hand-block
  the joint while polling ``Status`` and ``ping``, then restore both
  registers in a ``finally`` (they're EPROM-resident, so restore is
  mandatory).
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, field

from lerobot.motors.locked_bus import LockedBus
from lerobot.motors.motors_bus import SerialMotorsBus, get_address
from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    """Outcome of a non-physical recovery attempt on a motor bus.

    Attributes:
        port: Bus port the report describes. ``None`` when no bus was
            available (e.g. recovering a gRPC- or sim-backed robot).
        responsive_ids: Motor IDs that replied to the initial ping pass.
        torque_disabled_ids: IDs whose torque was successfully disabled
            during recovery.
        recovered_ids: Motor IDs that did not respond initially but came
            back after the driver-specific recovery trick.
        still_unreachable_ids: Motor IDs that never answered. These need
            a power cycle (Feetech) or hardware attention.
        errors: Per-ID error messages encountered during recovery.
        notes: Free-form messages (e.g. ``"could not open port: ..."``,
            ``"no buses found"``).
    """

    port: str | None = None
    responsive_ids: list[int] = field(default_factory=list)
    torque_disabled_ids: list[int] = field(default_factory=list)
    recovered_ids: list[int] = field(default_factory=list)
    still_unreachable_ids: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "responsive_ids": list(self.responsive_ids),
            "torque_disabled_ids": list(self.torque_disabled_ids),
            "recovered_ids": list(self.recovered_ids),
            "still_unreachable_ids": list(self.still_unreachable_ids),
            "errors": {str(k): v for k, v in self.errors.items()},
            "notes": list(self.notes),
        }


def recover_bus(bus: SerialMotorsBus, num_retry: int = 3) -> RecoveryReport:
    """Attempt non-physical recovery on one motor bus.

    The bus must not be currently connected by another caller — this function
    opens the port itself in order to bypass the strict handshake.

    Args:
        bus: The bus to recover.
        num_retry: Per-motor ping/write retry count. Defaults to 3.

    Returns:
        ``RecoveryReport`` describing what was found and what was done.
    """
    report = RecoveryReport(port=bus.port)
    try:
        bus._connect(handshake=False)
    except Exception as e:
        report.notes.append(f"could not open port: {e}")
        return report

    try:
        for name, motor in bus.motors.items():
            if bus.ping(motor.id, num_retry=num_retry) is None:
                continue
            report.responsive_ids.append(motor.id)
            try:
                bus.disable_torque(motors=[name], num_retry=num_retry)
                report.torque_disabled_ids.append(motor.id)
            except Exception as e:
                report.errors[motor.id] = f"disable_torque failed: {e}"

        _recover_non_responders(bus, report, num_retry=num_retry)

        for motor in bus.motors.values():
            if motor.id not in report.responsive_ids and motor.id not in report.recovered_ids:
                report.still_unreachable_ids.append(motor.id)
    finally:
        with suppress(Exception):
            bus.port_handler.closePort()

    return report


def _recover_non_responders(bus: SerialMotorsBus, report: RecoveryReport, num_retry: int) -> None:
    """Driver-specific second pass for motors that didn't reply initially.

    Adds successfully-recovered IDs to ``report.recovered_ids`` and
    ``report.torque_disabled_ids``.
    """
    # Late import keeps optional Feetech SDK off the critical path of this
    # module — recovery is a rarely-imported branch and we don't want it to
    # pull in scservo_sdk just for an isinstance check.
    is_feetech = False
    try:
        from lerobot.motors.feetech import FeetechMotorsBus

        is_feetech = isinstance(bus, FeetechMotorsBus)
    except ImportError:
        pass

    if is_feetech:
        _recover_feetech_non_responders(bus, report, num_retry=num_retry)
        return

    # Other drivers: nothing extra to do. Future Dynamixel support can call
    # ``bus.reboot(id)`` here when added.


def _recover_feetech_non_responders(bus: SerialMotorsBus, report: RecoveryReport, num_retry: int) -> None:
    """Clear Feetech STS/SMS overload latches via direct ``Torque_Enable=0``.

    STS/SMS motors latch into a "protected" state on overload/overcurrent.
    While latched, the motor either responds to broadcast PING with the
    Status error byte set (which the SDK treats as a missing reply) or
    appears entirely silent for a few packet cycles. Writing
    ``Torque_Enable=0`` directly is enough to acknowledge the unloaded
    condition and clear the latch on most firmware revisions; combined with
    a couple of retried pings (which give the protection circuit time to
    release) this recovers most overload-trip cases without a power cycle.
    A truly hung MCU stays silent and ends up in ``still_unreachable_ids``.
    """
    # Local import to avoid a top-level cycle / mandatory dep on the SDK.
    from lerobot.motors.feetech.feetech import TorqueMode

    for name, motor in bus.motors.items():
        if motor.id in report.responsive_ids:
            continue

        torque_addr, torque_len = get_address(bus.model_ctrl_table, motor.model, "Torque_Enable")
        bus._write(
            torque_addr,
            torque_len,
            motor.id,
            TorqueMode.DISABLED.value,
            num_retry=num_retry,
            raise_on_error=False,
        )

        if bus.ping(motor.id, num_retry=num_retry) is None:
            continue

        report.recovered_ids.append(motor.id)
        try:
            bus.disable_torque(motors=[name], num_retry=num_retry)
            report.torque_disabled_ids.append(motor.id)
        except Exception as e:
            report.errors[motor.id] = f"disable_torque after recover failed: {e}"


def recover_robot(robot: Robot, num_retry: int = 3) -> list[RecoveryReport]:
    """Walk a robot's attributes for buses / nested robots and recover each.

    Single-arm robots usually expose ``self.bus``; bi-arm composites expose
    ``self.left_arm`` / ``self.right_arm`` (each a ``Robot`` with its own
    ``bus``). Robots with no underlying ``SerialMotorsBus`` (gRPC, sim) get
    a single empty report with a "no buses found" note rather than an
    exception, so callers can treat the result uniformly.

    The robot must not currently hold any of its serial ports open.

    Args:
        robot: The robot to recover.
        num_retry: Forwarded to each ``recover_bus`` call. Defaults to 3.

    Returns:
        List of ``RecoveryReport``, one per recovered bus. Never empty: if no
        buses are found, returns a single placeholder report with a note.
    """
    reports: list[RecoveryReport] = []
    _walk(robot, reports, num_retry=num_retry, seen=set())

    if not reports:
        empty = RecoveryReport()
        empty.notes.append(f"no MotorsBus instances found on {type(robot).__name__}")
        reports.append(empty)
    return reports


def _walk(
    obj: object,
    reports: list[RecoveryReport],
    num_retry: int,
    seen: set[int],
) -> None:
    """Recurse into Robot attributes that hold buses or nested robots.

    Tracks ``id(obj)`` in ``seen`` to defend against cycles in case a
    custom robot subclass stores a back-reference to itself.
    """
    if id(obj) in seen:
        return
    seen.add(id(obj))

    for attr in vars(obj).values():
        # Unwrap LockedBus proxies — the predictive robots wrap their
        # FeetechMotorsBus in LockedBus for thread-safety, but the
        # recovery routine needs the underlying SerialMotorsBus directly
        # (recover_bus opens/closes the port and bypasses the proxy's
        # locking, since recovery runs while the bus is otherwise idle).
        if isinstance(attr, LockedBus):
            attr = attr._bus  # noqa: SLF001 — recovery is the LockedBus owner's peer
        if isinstance(attr, SerialMotorsBus):
            reports.append(recover_bus(attr, num_retry=num_retry))
        elif isinstance(attr, Robot):
            _walk(attr, reports, num_retry=num_retry, seen=seen)
