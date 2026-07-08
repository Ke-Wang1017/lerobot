#!/usr/bin/env python

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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate / DAgger a policy against an EXTERNAL robot — sim or real.

This repository ships only the abstract robot interface: concrete robot, motor,
and camera drivers live in external plugin packages and are loaded at runtime via
``make_robot_from_config`` / ``make_teleoperator_from_config``. Two ways to point
LeRobot at a robot for policy validation:

1. Simulation via the gym-env interface (no robot driver needed). LeRobot ships the
   ``aloha`` env (``AlohaEnv``), backed by the ``gym-aloha`` package:
   https://github.com/huggingface/gym-aloha — install with ``pip install 'lerobot[aloha]'``.
   This is the ``--mode sim`` path below and is what ``lerobot-eval --env.type=aloha``
   uses under the hood.

2. A real or simulated robot exposed as a ``Robot`` plugin. Install a robot-driver
   package that registers a ``RobotConfig`` subclass (e.g. ``my_aloha_plugin`` exposing
   ``--robot.type=aloha_bimanual``); ``make_robot_from_config`` resolves it through the
   plugin loader. This is the ``--mode robot`` path below; the GUI (``lerobot-gui``) and
   ``lerobot-record`` drive the exact same interface, so they work against the same plugin.

DAgger: a human corrects the policy via a teleop *intervention* device. Only
device-agnostic human-input teleoperators ship in-tree (``keyboard``, ``gamepad``,
``quest_vr``); this example uses ``keyboard``.

Examples
--------
    # Validate a policy in the aloha simulation (needs lerobot[aloha]):
    python examples/dagger_sim_aloha.py --mode sim --policy-path lerobot/act_aloha_sim_transfer

    # DAgger against an external robot plugin, keyboard intervention:
    python examples/dagger_sim_aloha.py --mode robot --robot-type aloha_bimanual \
        --policy-path lerobot/act_aloha_sim_transfer
"""

import argparse


def run_sim(policy_path: str, n_steps: int) -> None:
    """Validate a policy against the aloha simulation via the gym-env interface."""
    import gymnasium as gym  # noqa: F401  (imported by make_env)

    from lerobot.envs import AlohaEnv
    from lerobot.envs.factory import make_env

    # AlohaEnv is a registered EnvConfig ("aloha"); it needs no in-tree robot driver.
    env_cfg = AlohaEnv(task="AlohaInsertion-v0")
    envs = make_env(env_cfg, n_envs=1)
    print(f"Built aloha sim env(s): {list(envs)} — this is the sim-validation path.")

    # A trained policy would be loaded here and rolled out. See lerobot-eval /
    # src/lerobot/scripts/lerobot_eval.py for the full evaluation loop.
    #
    #   from lerobot.policies import make_policy
    #   policy = make_policy(...)  # from policy_path
    #   run the standard eval rollout over `envs`
    print(
        f"Load the policy from {policy_path!r} and roll out for {n_steps} steps "
        "(see lerobot-eval for the full rollout)."
    )


def run_robot(robot_type: str, teleop_type: str, policy_path: str) -> None:
    """DAgger loop skeleton against an external robot plugin, through the interface."""
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.config import TeleoperatorConfig

    # The concrete robot lives in an external plugin package; the plugin loader
    # resolves `robot_type` to its Robot implementation. No in-tree driver required.
    robot_cfg = RobotConfig.get_choice_class(robot_type)()  # type: ignore[call-arg]
    robot = make_robot_from_config(robot_cfg)

    # Human intervention device (device-agnostic; ships in-tree).
    teleop_cfg = TeleoperatorConfig.get_choice_class(teleop_type)()  # type: ignore[call-arg]
    teleop = make_teleoperator_from_config(teleop_cfg)

    print(
        f"Ready to DAgger: robot={type(robot).__name__} (plugin '{robot_type}'), "
        f"intervention={type(teleop).__name__}, policy={policy_path!r}.\n"
        "Loop: robot.get_observation() -> policy action, override with teleop when the "
        "human intervenes, robot.send_action(), log corrected transitions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["sim", "robot"], default="sim", help="Validation target.")
    parser.add_argument("--policy-path", default="lerobot/act_aloha_sim_transfer", help="Policy repo id or path.")
    parser.add_argument("--n-steps", type=int, default=400, help="Rollout length for --mode sim.")
    parser.add_argument("--robot-type", default="aloha_bimanual", help="External Robot plugin type (--mode robot).")
    parser.add_argument("--teleop-type", default="keyboard", help="Intervention teleop type (--mode robot).")
    args = parser.parse_args()

    if args.mode == "sim":
        run_sim(args.policy_path, args.n_steps)
    else:
        run_robot(args.robot_type, args.teleop_type, args.policy_path)


if __name__ == "__main__":
    main()
