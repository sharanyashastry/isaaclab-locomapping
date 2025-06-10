# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Teleop imports
import pygame

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--teleop",
    action="store_true",
    default=False,
    help="If set, read from joystick for teleop instead of using the predetermined velocity targets.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

def customize_env(env_cfg):
    """This function overrides the settings in SpotFlatEnvCfg with customizations. """
    import isaaclab.terrains as terrain_gen
    # print("terrain generator values are ", env_cfg.scene.terrain.terrain_generator)
    tg = env_cfg.scene.terrain.terrain_generator
    # Changing terrain sizing (The documentation says this is in meters)
    # Either this is not in meters or the other stuff is not in meters.
    tg.size = (1.0, 1.0)
    tg.num_rows = 2
    tg.num_cols = 2

    # turn the generator into a pure flat-plane creator
    tg.terrain_type      = "flat"
    tg.class_type        = "flat"        # older builds use this key
    tg.sub_terrains      = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0, size=tg.size )}

    # Drop the "time_out" termination so that the robot doesn't get reset every 20 s
    env_cfg.terminations.time_out = None
    env_cfg.is_finite_horizon = False            # tell the wrapper it's now infinite-horizon

    # Change init position of robot
    # env_cfg.scene.robot.init_state.pos = (0, 5.0, 0.9)
    # This is not in the environment frame. This is likely the base position in the body frame.
    # print("Robot position:", env_cfg.scene.robot.init_state.pos)
    # breakpoint()

import random
import math
from isaacsim.core.utils import prims            # USD helpers
from isaaclab.sim import (
    ConeCfg, MeshCuboidCfg,
    RigidBodyPropertiesCfg, MassPropertiesCfg,
    CollisionPropertiesCfg, PreviewSurfaceCfg,
)

def add_random_props(
    num_cones=10,
    num_cuboids=15,
    xy_range=(-10.0, 10.0),    # unsure if this is in meters
    height_above_ground=0.5, # spawn height fixed
    cone_radius=(0.2, 0.5),
    cone_height=(0.6, 1.2),
    cuboid_size=((0.2, 0.4), (0.2, 0.6), (0.2, 0.4))
):
    """
    Sprinkles random cones and cuboids under /World/envs/env_0.
    Must be called immediately after `env = gym.make(...)` and before any `env.step()`.
    """
    ENV0 = "/World/envs/env_0"
    prims.create_prim(f"{ENV0}/Props", "Xform")

    # color palette
    palette = [
        (1.0, 0.4, 0.4),
        (0.4, 1.0, 0.4),
        (0.4, 0.6, 1.0),
        (1.0, 0.65, 0.0),
        (0.8, 0.2, 1.0),
    ]

    def rand_xy():
        return (random.uniform(*xy_range),
                random.uniform(*xy_range))

    # spawn cones
    for i in range(num_cones):
        r   = random.uniform(*cone_radius)
        h   = random.uniform(*cone_height)
        x,y = rand_xy()
        yaw = random.uniform(-math.pi, math.pi)
        cfg = ConeCfg(
            radius=r, height=h,
            rigid_props   = RigidBodyPropertiesCfg(),
            mass_props    = MassPropertiesCfg(mass=1.0),
            collision_props = CollisionPropertiesCfg(),
            visual_material = PreviewSurfaceCfg(diffuse_color=random.choice(palette)),
        )
        cfg.func(
            f"{ENV0}/Props/Cone_{i}",
            cfg,
            translation=(x, y, height_above_ground),
            orientation=(math.cos(yaw/2), 0, 0, math.sin(yaw/2))
        )

    # spawn cuboids
    for i in range(num_cuboids):
        sx = random.uniform(*cuboid_size[0])
        sy = random.uniform(*cuboid_size[1])
        sz = random.uniform(*cuboid_size[2])
        x,y = rand_xy()
        yaw = random.uniform(-math.pi, math.pi)
        cfg = MeshCuboidCfg(
            size=(sx, sy, sz),
            rigid_props   = RigidBodyPropertiesCfg(),
            mass_props    = MassPropertiesCfg(mass=1.0),
            collision_props = CollisionPropertiesCfg(),
            visual_material = PreviewSurfaceCfg(diffuse_color=random.choice(palette)),
        )
        cfg.func(
            f"{ENV0}/Props/Cuboid_{i}",
            cfg,
            translation=(x, y, height_above_ground),
            orientation=(math.cos(yaw/2), 0, 0, math.sin(yaw/2))
        )

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    # WARNING: THESE IMPORTS HAVE TO STAY BEFORE THE gym.make but before the env.step to prevent errors.
    # Imports to add objects into the scene
    customize_env(env_cfg)
    add_random_props()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    pygame.init()

    count = pygame.joystick.get_count()
    print(f"[JOYSTICK] Detected {count} device(s).")

    if (count != 1):
        raise RuntimeError("Please connect exactly one controller")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Poll pygame events so joystick state updates
        if args_cli.teleop:
            pygame.event.pump()
            term = env.unwrapped.command_manager.get_term("base_velocity")  # UniformVelocityCommand

            # scale joystick commands
            lin_x_scaling = 3.0
            lin_y_scaling = 3.0
            ang_w_scaling = 2.0
            vx = lin_x_scaling * (-joystick.get_axis(1))   # forward/back (1.5)
            vy = lin_y_scaling * ( joystick.get_axis(0))   # left/right (1.5)
            wz = ang_w_scaling * ( -joystick.get_axis(2))   # yaw rate (1.2)

            # write directly into the term’s tensor (shape = [N, 3])
            term.command[:] = torch.tensor([vx, vy, wz],
                                        dtype=torch.float32, device=env.unwrapped.device).repeat(env.num_envs, 1)

            # fresh obs now contains the new [vx,vy,wz]
            obs, _ = env.get_observations()

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        else:
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
