#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
This script helps identify which SO107 robot arm is connected to which port
for bimanual teleoperation setup.

It will test each port (/dev/ttyACM0-3), move the base motor, and ask you to
identify which arm moved (LL=left leader, RL=right leader, LF=left follower, RF=right follower).

Then it outputs the correct port configuration for your lerobot-teleoperate command.
"""

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def test_port(port: str):
    """
    Test if an SO107 robot is connected to the given port by moving its base motor.

    Returns True if successful, False if failed.
    """
    # Define SO107 motors (7-DOF arm)
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "forearm_roll": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
    }

    try:
        print(f"\nTesting {port}...")
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        # Read current position
        current_pos = bus.read("Present_Position", "shoulder_pan", normalize=False)

        # Move the base motor to identify the robot
        print("Moving base motor... WATCH YOUR ROBOTS!")
        movement_ticks = 200

        # Move right
        bus.write("Goal_Position", "shoulder_pan", current_pos + movement_ticks, normalize=False)
        time.sleep(0.8)

        # Move left
        bus.write("Goal_Position", "shoulder_pan", current_pos - movement_ticks, normalize=False)
        time.sleep(0.8)

        # Return to center
        bus.write("Goal_Position", "shoulder_pan", current_pos, normalize=False)
        time.sleep(0.3)

        bus.disconnect()
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("LeRobot Bimanual SO107 Port Identifier")
    print("=" * 60)
    print("\nThis script will test ports /dev/ttyACM0-3")
    print("Watch which arm moves, then identify it:")
    print("  LL = Left Leader")
    print("  RL = Right Leader")
    print("  LF = Left Follower")
    print("  RF = Right Follower")
    print("=" * 60)

    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3"]
    port_mapping = {}

    for port in ports:
        if test_port(port):
            while True:
                response = input("\nWhich arm moved? [LL/RL/LF/RF] (or 'skip'): ").strip().upper()
                if response in ["LL", "RL", "LF", "RF"]:
                    port_mapping[response] = port
                    print(f"  ✓ Recorded: {response} -> {port}")
                    break
                elif response == "SKIP":
                    print("  Skipped")
                    break
                else:
                    print("  Invalid input. Please enter LL, RL, LF, RF, or skip")

    # Generate the command arguments
    print("\n" + "=" * 60)
    print("PORT CONFIGURATION")
    print("=" * 60)

    if len(port_mapping) == 4:
        print("\nAdd these arguments to your lerobot-teleoperate command:\n")
        print(f"  --robot.left_arm_port={port_mapping.get('LF', 'MISSING')} \\")
        print(f"  --robot.right_arm_port={port_mapping.get('RF', 'MISSING')} \\")
        print(f"  --teleop.left_arm_port={port_mapping.get('LL', 'MISSING')} \\")
        print(f"  --teleop.right_arm_port={port_mapping.get('RL', 'MISSING')} \\")
    else:
        print("\nWarning: Not all arms identified. Here's what we found:")
        for arm_type, port in sorted(port_mapping.items()):
            arm_name = {
                "LF": "robot.left_arm_port (Left Follower)",
                "RF": "robot.right_arm_port (Right Follower)",
                "LL": "teleop.left_arm_port (Left Leader)",
                "RL": "teleop.right_arm_port (Right Leader)",
            }
            print(f"  {arm_name.get(arm_type, arm_type)}: {port}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
