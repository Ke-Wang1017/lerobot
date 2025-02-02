from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.robot_devices.robots.piper import PiperRobot
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

inference_time_s = 60
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "lerobot/policy/piper_real_1/pretrained_model"
policy = DiffusionPolicy.from_pretrained(ckpt_path, local_files_only=False)
policy.to(device)
robot_config_path = 'lerobot/configs/robot/piper.yaml'
robot_cfg = init_hydra_config(robot_config_path)
robot = make_robot(robot_cfg)
robot.connect()


for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)
    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    state = robot.get_state()
    state = state["state"]
    state[3:6] = robot.euler_filter.rectify(state[3:6])
    # action[:6] += state[:6]
    # breakpoint()
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    print(f"dt_s: {dt_s}")
    # breakpoint()
    busy_wait(1 / fps - dt_s)