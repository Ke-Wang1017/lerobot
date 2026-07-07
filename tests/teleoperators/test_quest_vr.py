#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Property tests for the Quest VR teleoperator's pure-logic surface.

No Quest hardware, no HTTPS server: drives ``QuestArmController`` directly
with synthetic WebXR frames and asserts the properties that justify Quest
VR being usable as an intent source for a Cartesian-IK robot. Properties
covered (matching the proof checklist sketched in the PR body):

1. **Engage-relative deltas.** ``target_xyz`` is the position delta from
   the engage snapshot, scaled by ``position_scale``. ``rotvec`` is the
   orientation delta from the engage snapshot.
2. **No drift on hold.** Clutch held + hand held still = ``target_xyz``
   stays at zero across many WebXR frames.
3. **Glitch suppression.** A jump larger than ``max_pos_step_m_per_tick``
   between consecutive Quest poses is clipped before it enters the delta.
4. **Bimanual independence.** A frame with only one hand updates that
   arm's action keys and routes the other hand to tracking-lost (its
   action keys stay disengaged).
5. **Gripper trigger mapping.** Engaged + trigger=t maps motor pos
   linearly from open to closed. Disengaged frames hold the last command.
6. **Tracking-dropout re-anchor.** After a frame with no pose for this
   controller, the next tracked frame re-anchors engage state — so a
   long no-tracking gap can never leak into ``target_xyz`` as a slew.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.teleoperators.quest_vr.arm_controller import QuestArmController
from lerobot.teleoperators.quest_vr.configuration_quest_vr import QuestVRTeleopConfig
from lerobot.teleoperators.quest_vr.server import (
    QUEST_TO_ROBOT_M,
    quest_delta_to_robot,
)

# ── Helpers ───────────────────────────────────────────────────────────────


_CLUTCH_IDX = 1  # default
_TRIGGER_IDX = 0  # default
_OPEN_MOTOR = 0.0
_CLOSED_MOTOR = 100.0
_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def _make_controller(**overrides) -> QuestArmController:
    """Default controller params — identity scale, generous caps for unit tests."""
    defaults = {
        "clutch_button_index": _CLUTCH_IDX,
        "gripper_button_index": _TRIGGER_IDX,
        "position_scale": 1.0,
        "max_rot_step_rad_per_tick": np.pi,
        "max_pos_step_m_per_tick": 1.0,  # off by default — tests opt in
        "gripper_open_motor": _OPEN_MOTOR,
        "gripper_closed_motor": _CLOSED_MOTOR,
        "key_prefix": "",
    }
    defaults.update(overrides)
    return QuestArmController(**defaults)


_RESET_IDX = 4  # default — A/X face button (see QuestVRTeleopConfig.reset_button_index)


def _pose(pos, *, clutch: float = 1.0, trigger: float = 0.0, reset: float = 0.0, rot=_IDENTITY_QUAT):
    """Build a WebXR-shaped pose dict for one controller."""
    buttons = [0.0] * (max(_RESET_IDX, _TRIGGER_IDX, _CLUTCH_IDX) + 1)
    buttons[_TRIGGER_IDX] = float(trigger)
    buttons[_CLUTCH_IDX] = float(clutch)
    buttons[_RESET_IDX] = float(reset)
    return {"pos": list(pos), "rot": list(rot), "buttons": buttons}


# ── 1. Engage-relative deltas ─────────────────────────────────────────────


def test_disengaged_action_is_idle_and_keeps_gripper_open():
    ctrl = _make_controller()
    action = ctrl.process_pose(_pose([0.1, 0.2, 0.3], clutch=0.0, trigger=0.5))

    assert action["enabled"] == 0.0
    for k in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"):
        assert action[k] == 0.0, f"{k} should be zero while disengaged"
    # Trigger touch while disengaged should NOT slam the gripper.
    assert action["gripper_pos"] == _OPEN_MOTOR


def test_engaged_targets_are_position_delta_from_engage_snapshot():
    ctrl = _make_controller()
    # Engage at a non-trivial Quest position.
    engage = np.array([0.5, 1.2, -0.3])
    a0 = ctrl.process_pose(_pose(engage, clutch=1.0))
    assert a0["enabled"] == 1.0
    # First engaged frame: hand hasn't moved since engage, so zero delta.
    assert pytest.approx(a0["target_x"], abs=1e-12) == 0.0
    assert pytest.approx(a0["target_y"], abs=1e-12) == 0.0
    assert pytest.approx(a0["target_z"], abs=1e-12) == 0.0

    # Move the hand 7 cm along Quest +x. target_xyz is QUEST_TO_ROBOT_M @ delta.
    dquest = np.array([0.07, 0.0, 0.0])
    a1 = ctrl.process_pose(_pose(engage + dquest, clutch=1.0))
    drobot_expected = quest_delta_to_robot(dquest)
    np.testing.assert_allclose(
        [a1["target_x"], a1["target_y"], a1["target_z"]],
        drobot_expected,
        atol=1e-12,
    )


def test_position_scale_multiplies_target_delta():
    ctrl = _make_controller(position_scale=0.5)
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))  # engage
    a = ctrl.process_pose(_pose([0.10, 0.0, 0.0], clutch=1.0))

    drobot = quest_delta_to_robot(np.array([0.10, 0.0, 0.0])) * 0.5
    np.testing.assert_allclose([a["target_x"], a["target_y"], a["target_z"]], drobot, atol=1e-12)


def test_axis_mapping_is_consistent_with_quest_to_robot_matrix():
    """The robot-frame delta for each Quest unit vector equals that column
    of QUEST_TO_ROBOT_M (sanity-checks the frame conversion is wired
    through with no surprise sign flips)."""
    for col, quest_axis in enumerate(np.eye(3)):
        ctrl = _make_controller()
        ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))
        a = ctrl.process_pose(_pose(quest_axis, clutch=1.0))
        np.testing.assert_allclose(
            [a["target_x"], a["target_y"], a["target_z"]],
            QUEST_TO_ROBOT_M[:, col],
            atol=1e-12,
            err_msg=f"Quest axis {col} mapping",
        )


# ── 2. No drift on hold ───────────────────────────────────────────────────


def test_held_clutch_with_still_hand_does_not_drift():
    ctrl = _make_controller()
    ctrl.process_pose(_pose([0.1, 0.2, 0.3], clutch=1.0))  # engage

    # 100 identical WebXR frames at the same Quest pose. target_xyz must
    # stay exactly zero — any drift here would slew the IK target.
    for _ in range(100):
        a = ctrl.process_pose(_pose([0.1, 0.2, 0.3], clutch=1.0))
        assert a["enabled"] == 1.0
        for k in ("target_x", "target_y", "target_z"):
            assert a[k] == 0.0, f"{k} drifted while clutch held + hand still"


# ── 3. Glitch suppression ─────────────────────────────────────────────────


def test_glitch_jump_is_clipped_to_max_pos_step_per_tick():
    cap = 0.02  # 2 cm/frame
    ctrl = _make_controller(max_pos_step_m_per_tick=cap)
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))  # engage at origin

    # 50 cm jump in one frame — clearly a tracking-teleport glitch.
    a = ctrl.process_pose(_pose([0.5, 0.0, 0.0], clutch=1.0))

    # |target_xyz| must be ≤ cap (with position_scale=1 the robot-frame
    # delta has the same magnitude as the Quest-frame step after clipping).
    mag = float(np.linalg.norm([a["target_x"], a["target_y"], a["target_z"]]))
    assert mag <= cap + 1e-9, f"|target| {mag} exceeded cap {cap} — glitch leaked"


def test_glitch_cap_off_when_set_to_zero():
    ctrl = _make_controller(max_pos_step_m_per_tick=0.0)  # off
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))
    a = ctrl.process_pose(_pose([0.5, 0.0, 0.0], clutch=1.0))

    mag = float(np.linalg.norm([a["target_x"], a["target_y"], a["target_z"]]))
    assert mag > 0.4, "0.0 cap should disable glitch suppression entirely"


# ── 4. Per-arm independence (via the teleop's frame callback) ────────────


def test_only_left_hand_updates_only_left_keys():
    cfg = QuestVRTeleopConfig(
        left_gripper_open_motor=_OPEN_MOTOR,
        left_gripper_closed_motor=_CLOSED_MOTOR,
        right_gripper_open_motor=_OPEN_MOTOR,
        right_gripper_closed_motor=_CLOSED_MOTOR,
    )
    # Build the teleop instance without going through connect() — we only
    # need the _on_frame path, no HTTPS server.
    from lerobot.teleoperators.quest_vr.teleop_quest_vr import QuestVRTeleop

    teleop = QuestVRTeleop(cfg)
    # Engage left, leave right tracking-lost (no entry in poses[]).
    teleop._on_frame({"poses": [{"hand": "left", **_pose([0.0, 0.0, 0.0], clutch=1.0)}]})
    teleop._on_frame({"poses": [{"hand": "left", **_pose([0.1, 0.0, 0.0], clutch=1.0)}]})

    action = teleop.get_action_raw()
    # Left engaged + moved.
    assert action["left_enabled"] == 1.0
    assert any(action[k] != 0.0 for k in ("left_target_x", "left_target_y", "left_target_z"))
    # Right untouched — never tracked, never engaged.
    assert action["right_enabled"] == 0.0
    for k in ("right_target_x", "right_target_y", "right_target_z"):
        assert action[k] == 0.0


def test_action_features_has_all_per_arm_prefixed_keys():
    cfg = QuestVRTeleopConfig()
    from lerobot.teleoperators.quest_vr.teleop_quest_vr import (
        _PER_ARM_KEYS,
        ACTION_KEYS,
        QuestVRTeleop,
    )

    teleop = QuestVRTeleop(cfg)
    feats = teleop.action_features
    assert set(feats["names"]) == set(ACTION_KEYS)
    assert len(feats["names"]) == 2 * len(_PER_ARM_KEYS)  # left + right copy each key


# ── 5. Gripper trigger mapping ────────────────────────────────────────────


def test_gripper_pos_linearly_maps_trigger_when_engaged():
    ctrl = _make_controller()
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))  # engage

    # trigger=0 → open, trigger=0.5 → midpoint, trigger=1 → closed.
    for trig, expected in [(0.0, 0.0), (0.5, 50.0), (1.0, 100.0)]:
        a = ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0, trigger=trig))
        assert pytest.approx(a["gripper_pos"], abs=1e-9) == expected, f"trigger={trig}"


def test_disengaged_holds_last_gripper_command():
    ctrl = _make_controller()
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0, trigger=0.0))  # engage open
    a_closed = ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0, trigger=1.0))
    assert a_closed["gripper_pos"] == _CLOSED_MOTOR

    # Release clutch, wiggle trigger — gripper holds the last engaged command.
    a_held = ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=0.0, trigger=0.0))
    assert a_held["enabled"] == 0.0
    assert a_held["gripper_pos"] == _CLOSED_MOTOR


# ── 6. Tracking-dropout re-anchor ─────────────────────────────────────────


def test_tracking_lost_then_reacquire_reanchors_no_slew():
    """The whole point of on_tracking_lost: a controller that drops out of
    tracking while clutch is held cannot, on re-acquire, leak the moved-while-
    untracked offset into ``target_xyz`` as a slew. The re-acquire frame is a
    fresh rising edge."""
    ctrl = _make_controller()
    # Engage at origin, move 1 cm — this should produce a small delta.
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))
    a_moved = ctrl.process_pose(_pose([0.01, 0.0, 0.0], clutch=1.0))
    assert float(np.linalg.norm([a_moved["target_x"], a_moved["target_y"], a_moved["target_z"]])) > 0.005

    # Tracking lost for one frame (controller occluded / asleep).
    idle = ctrl.on_tracking_lost()
    assert idle["enabled"] == 0.0

    # User physically walks 50 cm with controller off-tracked. On the next
    # tracked frame the clutch is still held. Must NOT produce a 50 cm delta:
    # the engage snapshot was dropped, so this re-engages at the current pose.
    a_reanchor = ctrl.process_pose(_pose([0.5, 0.0, 0.0], clutch=1.0))
    mag = float(np.linalg.norm([a_reanchor["target_x"], a_reanchor["target_y"], a_reanchor["target_z"]]))
    assert mag < 1e-9, f"|target| {mag} after re-acquire — engage snapshot was not dropped on tracking-lost"


# ── 7. Reset re-anchor while clutch held ──────────────────────────────────


def test_reset_held_with_clutch_held_reanchors_engage_snapshot():
    """If the user holds the reset (A/X) button without releasing the
    clutch (grip), the IK side ramps the arm toward q_init while the
    Quest side previously kept the *original* engage snapshot. The
    moment reset releases, the next non-reset frame applied the full
    accumulated ``quest_now - quest_at_engage`` delta to the now-reset
    arm pose — a 20-30 cm hand drift turned into a sudden EE jump.

    Fix re-anchors the engage snapshot every reset frame while clutch
    is held, so ``target_xyz`` is zero across the reset-to-release
    transition.
    """
    ctrl = _make_controller()
    # Engage at origin (clutch press, no reset).
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))

    # Operator moves hand 30 cm while clutch + reset are BOTH held.
    # IK is ramping the arm during these frames; we don't care what
    # target_xyz reports while reset is held (the IK ignores it).
    ctrl.process_pose(_pose([0.1, 0.0, 0.0], clutch=1.0, reset=1.0))
    ctrl.process_pose(_pose([0.2, 0.0, 0.0], clutch=1.0, reset=1.0))
    ctrl.process_pose(_pose([0.3, 0.0, 0.0], clutch=1.0, reset=1.0))

    # Operator releases reset, clutch still held. Hand is at [0.3, 0, 0]
    # — 30 cm from the original engage position. Without the fix
    # ``target_x`` would be 0.30 m (scaled). With the fix the engage
    # snapshot was re-anchored to the latest reset pose, so the
    # not-yet-moved next frame produces zero delta.
    a = ctrl.process_pose(_pose([0.3, 0.0, 0.0], clutch=1.0, reset=0.0))
    mag = float(np.linalg.norm([a["target_x"], a["target_y"], a["target_z"]]))
    assert mag < 1e-9, (
        f"|target| {mag} after releasing reset with clutch held — "
        "engage snapshot was not re-anchored during the hold"
    )

    # Subsequent motion is measured from the re-anchored snapshot:
    # moving 5 cm further produces a 5 cm delta, not 35 cm.
    a2 = ctrl.process_pose(_pose([0.35, 0.0, 0.0], clutch=1.0))
    delta_mag = float(np.linalg.norm([a2["target_x"], a2["target_y"], a2["target_z"]]))
    assert delta_mag == pytest.approx(0.05, abs=1e-6)


# ── Config sanity ─────────────────────────────────────────────────────────


def test_config_registers_under_quest_vr_type_string():
    assert QuestVRTeleopConfig().type == "quest_vr"


def test_served_html_substitutes_clutch_button_index():
    """The server replaces ``{{CLUTCH_BUTTON_INDEX}}`` with the configured
    index so the page-side haptic feedback fires on the right button. A
    template marker that leaks through to the browser would silently
    disable the feedback (parseInt(\"{{...}}\") -> NaN -> fallback 1)."""
    import socket

    from lerobot.teleoperators.quest_vr.teleop_quest_vr import QuestVRTeleop

    with socket.socket() as ss:
        ss.bind(("127.0.0.1", 0))
        port = ss.getsockname()[1]
    cfg = QuestVRTeleopConfig(port=port, clutch_button_index=3)
    teleop = QuestVRTeleop(cfg)
    teleop.connect()
    try:
        # Local-bound request to the just-started server. We disable cert
        # verification because the cert is self-signed at first run.
        import ssl as _ssl
        import urllib.request

        ctx = _ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        with urllib.request.urlopen(f"https://127.0.0.1:{port}/", context=ctx, timeout=5) as resp:
            html = resp.read().decode()
        # Marker substituted with the configured value, not leaking through.
        assert "{{CLUTCH_BUTTON_INDEX}}" not in html
        assert 'parseInt("3", 10)' in html
    finally:
        teleop.disconnect()


def test_server_pushes_hold_messages_on_state_change():
    """End-to-end check that ``QuestServer`` translates a per-arm IK-hold
    flag flip into a ``{type: "hold"}`` WS message — the back-channel the
    page-side rumble feedback rides on. No browser; the test is a
    direct-WS client driving the server.

    Verifies:
      * No ``hold`` on the first frame (state matches initial ``(False, False)``)
      * A ``hold`` message lands when the state flips
      * No duplicate ``hold`` while steady
    """
    import asyncio
    import socket
    import ssl as _ssl

    import aiohttp

    from lerobot.teleoperators.quest_vr.server import QuestServer
    from lerobot.teleoperators.quest_vr.teleop_quest_vr import CERT_DIR, HTML_PATH

    async def _drive():
        with socket.socket() as ss:
            ss.bind(("127.0.0.1", 0))
            port = ss.getsockname()[1]

        hold_state = {"value": (False, False)}
        server = QuestServer(
            html_path=HTML_PATH,
            port=port,
            cert_dir=CERT_DIR,
            on_frame=lambda _frame: None,
            get_hold_state=lambda: hold_state["value"],
        )
        server.start()
        try:
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ctx)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.ws_connect(f"wss://127.0.0.1:{port}/ws") as ws:

                    async def collect_for(seconds: float) -> list[dict]:
                        out: list[dict] = []
                        deadline = asyncio.get_event_loop().time() + seconds
                        while True:
                            remaining = deadline - asyncio.get_event_loop().time()
                            if remaining <= 0:
                                break
                            try:
                                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                            except TimeoutError:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                out.append(msg.json())
                        return out

                    # Frame 1: state still (False, False). No `hold` should fire.
                    await ws.send_json({"type": "frame", "t_quest_send": 0.0, "poses": []})
                    msgs = await collect_for(0.2)
                    assert not any(m.get("type") == "hold" for m in msgs), msgs

                    # Flip left to True, send a frame — expect exactly one hold message.
                    hold_state["value"] = (True, False)
                    await ws.send_json({"type": "frame", "t_quest_send": 0.0, "poses": []})
                    msgs = await collect_for(0.2)
                    holds = [m for m in msgs if m.get("type") == "hold"]
                    assert holds == [{"type": "hold", "left": True, "right": False}], holds

                    # Steady-state frame — no further hold message.
                    await ws.send_json({"type": "frame", "t_quest_send": 0.0, "poses": []})
                    msgs = await collect_for(0.2)
                    assert not any(m.get("type") == "hold" for m in msgs), msgs

                    # Flip right too — single transition message.
                    hold_state["value"] = (True, True)
                    await ws.send_json({"type": "frame", "t_quest_send": 0.0, "poses": []})
                    msgs = await collect_for(0.2)
                    holds = [m for m in msgs if m.get("type") == "hold"]
                    assert holds == [{"type": "hold", "left": True, "right": True}], holds

                    # Release both — single back-to-rest message.
                    hold_state["value"] = (False, False)
                    await ws.send_json({"type": "frame", "t_quest_send": 0.0, "poses": []})
                    msgs = await collect_for(0.2)
                    holds = [m for m in msgs if m.get("type") == "hold"]
                    assert holds == [{"type": "hold", "left": False, "right": False}], holds
        finally:
            server.stop()

    asyncio.run(_drive())


def test_config_clutch_and_gripper_button_defaults():
    cfg = QuestVRTeleopConfig()
    # Default Quest 3 mapping: grip (1) = clutch, trigger (0) = gripper.
    assert cfg.clutch_button_index == 1
    assert cfg.gripper_button_index == 0
