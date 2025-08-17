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

# Camera related imports
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_rgbd
from single_frame_semantic_seg import build_pointcloud_from_rgbd, canon, project_single_point_into_world
import omni.usd
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
import numpy as np

from scipy.spatial.transform import Rotation as R
def compute_transformation_matrix(position, quat_wxyz, device):
    """
    Compute the transformation matrix from the position and orientation.
    """
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]  # Convert wxyz to xyzw since scipy expects xyzw
    rotation_matrix_np = R.from_quat(quat_xyzw.cpu().numpy()).as_matrix()
    rotation_matrix_torch = torch.from_numpy(rotation_matrix_np).to(device, dtype=torch.float32)
    transformation_matrix = torch.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix_torch
    transformation_matrix[:3, 3] = position
    return transformation_matrix

def yaw_from_quat_wxyz(q):
    # q can be torch or np array [w, x, y, z]
    w, x, y, z = [float(v) for v in q]
    return R.from_quat([x, y, z, w]).as_euler('xyz')[2]

def quat_from_yaw(th):
    """th in radians. Returns [w, x, y, z]."""
    if torch.is_tensor(th):
        half = th * 0.5
        w = torch.cos(half)
        z = torch.sin(half)
        zeros = torch.zeros_like(th)
        return torch.stack((w, zeros, zeros, z), dim=-1)
    else:
        th = float(th)
        half = 0.5 * th
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


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
    tg.class_type        = "flat" 
    tg.sub_terrains      = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0, size=tg.size )}

    # Drop the "time_out" termination so that the robot doesn't get reset every 20 s
    env_cfg.terminations.time_out = None
    env_cfg.is_finite_horizon = False            # tell the wrapper it's now infinite-horizon
    env_cfg.terminations.body_contact = None  # disable body contact termination, this means the robot can fall over and not reset

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
import isaaclab.sim as sim_utils

from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import convert_camera_frame_orientation_convention
def add_a_fixed_camera_and_prop():

    ENV0 = "/World/envs/env_0"

    prims.create_prim(f"{ENV0}/FixedProps", "Xform")

    cfg = ConeCfg(
            radius=0.3,
            height=1.2,
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=1.0),
            collision_props=CollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 1.0)),
        )
    cfg.func(
        f"{ENV0}/FixedProps/Cone",
        cfg,
        translation=(2.5, 0.5, 0.5),
        orientation=(1, 0, 0, 0),
    )


    prims.create_prim(f"{ENV0}/FixedCameraProp", "Xform")
    
    # Create a cone that will be visible to the camera
    # Position it in front of the robot camera (assuming camera is at offset [0.3, 0.0, 0.2] from robot)
    # and robot starts at origin, so camera is roughly at [0.3, 0.0, 0.2]
    # Place the prop 1.5 meters in front of the camera
    prop_distance = 3.5  # meters in front of camera
    prop_position = (0.3 + prop_distance, 0.0, 0.2)  # x=forward, y=left, z=up
    
    cfg_red_cone = ConeCfg(
        radius=0.3,
        height=0.8,
        rigid_props=RigidBodyPropertiesCfg(),
        mass_props=MassPropertiesCfg(mass=1.0),
        collision_props=CollisionPropertiesCfg(),
        visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red color
    )
    cfg_red_cone.func(
        f"{ENV0}/FixedCameraProp/Cone",
        cfg_red_cone,
        translation=prop_position,
        orientation=(1, 0, 0, 0),  # Identity quaternion
    )
    
    print(f"[INFO] Added fixed red cone at position {prop_position} (in front of camera)")

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

# Publishing related imports
from rgbd_interfaces.msg import RGBDFrame
from tiamat.communication.ros_rgbd_pub import RGBDPublisherRunner
from tiamat.communication.waypoint2D_sub import Waypoint2DSub
from rclpy.executors import SingleThreadedExecutor
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

def send_stop():
    rclpy.init()
    node = rclpy.create_node("pipeline_stop_pub")
    pub  = node.create_publisher(Empty, "/pipeline/stop", 10)
    # give the graph a moment to connect
    msg = Empty()
    for _ in range(3):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

def main():
    """Play with RSL-RL agent."""
    print("[DEBUG] Main function starting...")
    
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
    print("[DEBUG] Customizing environment...")
    customize_env(env_cfg)
    print("[DEBUG] Adding props...")
    # add_random_props()
    add_a_fixed_camera_and_prop()
    
    print("[DEBUG] Creating gym environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)    
    print("[DEBUG] Gym environment created successfully.")

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
    print("[DEBUG] Exporting policy...")
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    print("[DEBUG] Resetting environment...")
    obs, _ = env.get_observations()
    timestep = 0

    pygame.init()

    # count = pygame.joystick.get_count()
    # print(f"[JOYSTICK] Detected {count} device(s).")

    # # Uncomment this when no one else is using the computer and you have only one joystick.
    # # if (count != 1):
    # #     raise RuntimeError("Please connect exactly one controller")
    # joystick = pygame.joystick.Joystick(0)
    # joystick.init()

    # Frame capture data
    frame_capture_data = {
        'frame_numbers': [],
        'simulation_times': [],
        'robot_positions': [],
        'camera_positions': [],
        'camera_orientations': [],
        'camera_intrinsics': [],
        'depth_frames': [],
        'rgb_frames': [],
        'pointcloud_in_world_frame': [],
        'points_in_world_color': [],
        'bounding_box_coordinates': []
    }
    fixed_cam_frame_capture_data = {
        'frame_numbers': [],
        'simulation_times': [],
        'robot_positions': [],
        'camera_positions': [],
        'camera_orientations': [],
        'camera_intrinsics': [],
        'depth_frames': [],
        'rgb_frames': [],
        'pointcloud_in_world_frame': [],
        'points_in_world_color': [],
        'bounding_box_coordinates': []
    }

    print("[DEBUG] Starting RGBD publisher...")
    # The runner for RGBD publisher internally sets the rclpy init. 
    # @sharanyashastry TODO: clean this up so there's no accidental double setting of rclpy init later.
    runner = RGBDPublisherRunner("/spot/front/rgbd_frame")
    pub = runner.start()   # starts ROS + executor thread


    wp_sub = Waypoint2DSub("/waypoint_cmd_2d")
    exec_ = SingleThreadedExecutor()
    exec_.add_node(wp_sub)
    thr = threading.Thread(target=exec_.spin, daemon=True)
    thr.start()
    
    print("[DEBUG] Starting simulation loop...")
    ii = 0

    # Set up latching switch for capturing frames.
    capture_frames = False
    num_nav_steps = 0
    # simulate environment
    while simulation_app.is_running():
        ii+=1
        # Check capture_frames state and update it based on joystick button X.
        # if args_cli.teleop and joystick.get_button(0):
        #     capture_frames = not capture_frames

        # Stop capturing frames when joystick button A is pressed and break out of the simulation loop.
        if ii >= 151 or num_nav_steps > 10: #args_cli.teleop and joystick.get_button(1):
            # Send stop signal here.
            try:
                send_stop()  # fire-and-forget Empty on /pipeline/stop
            except Exception as e:
                print(f"[WARN] failed to publish stop: {e}")
            break

        start_time = time.time()
        
        current_sim_time = env.unwrapped.sim.get_physics_dt() * ii
        print(f"[DEBUG] Current sim time: {current_sim_time}")
        
        # Get the assets
        robot = env.unwrapped.scene["robot"]
        camera = env.unwrapped.scene["camera"]
        fixed_camera = env.unwrapped.scene["fixed_camera"]
        # THIS LINE GIVES YOU THE QUATERNION ACCORDING TO WHAT IS CALLED THE "WORLD" CONVENTION!! 
        # THIS IS NOT THE SAME AS QUATERNION IN WORLD FRAME, Hence the conversion.
        fixed_camera_pos_w, fixed_camera_quat_w_wxyz = fixed_camera._view.get_world_poses()
        fixed_camera_quat_w_wxyz = convert_camera_frame_orientation_convention(
            fixed_camera_quat_w_wxyz, origin="opengl", target="ros"
        )
        print(f"[FIXED CAM DEBUG] fixed camera position before setting : {fixed_camera_pos_w}")
        print(f"[FIXED CAM DEBUG] fixed camera orientation before setting : {fixed_camera_quat_w_wxyz}")

        # Get the position and orientation of the assets
        robot_body_index  = robot.data.body_names.index("body")
        robot_position = robot.data.body_pos_w[:,robot_body_index,:].squeeze()
        robot_quat_wxyz = robot.data.body_quat_w[:,robot_body_index,:].squeeze()
        W_T_R = compute_transformation_matrix(robot_position, robot_quat_wxyz, device = env.unwrapped.device)
        # print(f"[DEBUG] W_T_R Transformation matrix: {W_T_R}")

        camera_offset_pos = torch.tensor([0.3, 0.0, 0.2], dtype=torch.float32, device=env.unwrapped.device)
        camera_offset_quat_wxyz = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)
        # NOTE: @sharanyashastry Verify that I don't need to convert to world convention first before doing this.
        R_T_C = compute_transformation_matrix(camera_offset_pos, camera_offset_quat_wxyz, device = env.unwrapped.device)
        # print(f"[DEBUG] R_T_C Transformation matrix: {R_T_C}")
        W_T_C = W_T_R @ R_T_C
        # print(f"[DEBUG] W_T_C Transformation matrix: {W_T_C}")

        # Extract position and orientation from the camera pose in world frame according
        # to the camera instantiated in the ROS convention.
        camera_pos_w = (W_T_C[:3, 3]).unsqueeze(0)
        camera_quat_w_xyzw = torch.tensor(R.from_matrix(W_T_C[:3, :3].cpu().numpy()).as_quat(),
            dtype=torch.float32, device=env.unwrapped.device
        )
        camera_quat_w_wxyz = camera_quat_w_xyzw[[3, 0, 1, 2]]
        # print(f"[DEBUG] camera position: {camera_pos_w}")
        # print(f"[DEBUG] camera orientation: {camera_quat_w_wxyz}")
        # Setting latest to None as signal to use default velocity commands.
        latest = None
        # Start capturing frames when joystick button X is pressed.
        if ii >= 20:  #args_cli.teleop and capture_frames:
            print(f"[DEBUG] Capturing frame {ii}...")
            frame_capture_data['frame_numbers'].append(ii)
            frame_capture_data['simulation_times'].append(current_sim_time)
            frame_capture_data['robot_positions'].append(robot_position)
            frame_capture_data['camera_positions'].append(camera_pos_w)
            frame_capture_data['camera_orientations'].append(camera_quat_w_wxyz)
            frame_capture_data['camera_intrinsics'].append(camera.data.intrinsic_matrices[0])

            # Capturing data for fixed camera.
            fixed_cam_frame_capture_data['frame_numbers'].append(ii)
            fixed_cam_frame_capture_data['simulation_times'].append(current_sim_time)
            fixed_cam_frame_capture_data['camera_positions'].append(fixed_camera_pos_w)
            print(f"[FIXED CAM DEBUG] Camera position : {fixed_camera_pos_w}")
            print(f"[FIXED CAM DEBUG] Camera orientation in ros convention : {fixed_camera_quat_w_wxyz}")
            # Convert the camera quaternion from ros to world convention.
            fixed_camera_quat_in_world_convention = convert_camera_frame_orientation_convention(
                fixed_camera_quat_w_wxyz, origin="ros", target="world"
            )
            print(f"[FIXED CAM DEBUG] Camera orientation in world convention : {fixed_camera_quat_in_world_convention}")
            fixed_cam_frame_capture_data['camera_orientations'].append(fixed_camera_quat_w_wxyz)
            fixed_cam_frame_capture_data['camera_intrinsics'].append(fixed_camera.data.intrinsic_matrices[0])
            fixed_camera_rgb_data = fixed_camera.data.output["rgb"][0].detach().cpu().clone()
            fixed_camera_depth_data = fixed_camera.data.output["distance_to_image_plane"][0].detach().cpu().clone()
            fixed_cam_frame_capture_data['depth_frames'].append(fixed_camera_depth_data)
            fixed_cam_frame_capture_data['rgb_frames'].append(fixed_camera_rgb_data)
            
            # Collect the rgb and depth frames from the camera AFTER position calculation
            rgb_data = camera.data.output["rgb"][0].detach().cpu().clone()
            depth_data = camera.data.output["distance_to_image_plane"][0].detach().cpu().clone()
            # print(f"[DEBUG] RGB data shape: {rgb_data.shape}")
            # print(f"[DEBUG] Depth data shape: {depth_data.shape}")

            frame_capture_data['depth_frames'].append(depth_data)
            frame_capture_data['rgb_frames'].append(rgb_data)

            rgb_np = rgb_data.detach().cpu().numpy()
            
            # Add publishing function here.
            pub.publish_frame(
                frame_idx    = ii,
                stamp_s      = float(current_sim_time),
                rgb          = fixed_camera_rgb_data,                               # (H,W,3) uint8
                depth        = fixed_camera_depth_data,                             # or None
                K_3x3        = fixed_camera.data.intrinsic_matrices[0],      # (3,3)
                cam_pos_w    = fixed_camera_pos_w.squeeze(),                 # (3,)
                cam_quat_wxyz= fixed_camera_quat_w_wxyz.squeeze(),          # (4,)
                frame_id     = "spot/front_cam_optical",
            )

            latest = wp_sub.get_latest()
            if latest is not None:
                x, y, th, stamp = latest
                print("Received next waypoint : ", x, y, th)
                x_t = torch.as_tensor(x, dtype = fixed_camera_pos_w.dtype, device = fixed_camera_pos_w.device)
                y_t = torch.as_tensor(y, dtype = fixed_camera_pos_w.dtype, device = fixed_camera_pos_w.device)
                th_t = torch.as_tensor(th, dtype = fixed_camera_quat_w_wxyz.dtype, device = fixed_camera_quat_w_wxyz.device)
                next_camera_quat = quat_from_yaw(th_t).reshape(1,4)
                print("After converting received yaw into a quat ", next_camera_quat)
                # convert quat to ros convention
                next_camera_quat = convert_camera_frame_orientation_convention(
                    next_camera_quat, origin="world", target="ros"
                )
                    # breakpoint()
                # Set convention = ros for set_world_poses if using the fixed quaternion in the next line.
                # camera_offset_quat_wxyz.clone().detach().reshape(1,4) #quat_from_yaw(th_t)
                next_camera_position = fixed_camera_pos_w.clone().detach()
                next_camera_position[0,0] = x_t
                next_camera_position[0,1] = y_t
                fixed_camera.set_world_poses(next_camera_position, next_camera_quat, convention="ros")
                num_nav_steps += 1
            # computing velocity command based on current position and next waypoint.
            # Using the camera position for now but this will need to change to robot position in the future.
            # Also, fix the shape of the camera position to be 3, and not 1,3
                error_x_world = x - camera_pos_w[0, 0]
                error_y_world = y - camera_pos_w[0, 1]
                curr_th_world = yaw_from_quat_wxyz(camera_quat_w_wxyz)
                error_th_world = th - curr_th_world
                # convert errors into base frame.
                W_T_R_np = W_T_R[:3, :3].detach().cpu().numpy() 
                error_in_robot_frame = W_T_R_np.T@np.array([error_x_world, error_y_world, 0.0], dtype=float)

            # # --- goal pose from waypoint
            # gx, gy, gth = float(x), float(y), float(th)

            # # --- current ROBOT pose in WORLD (use robot, not camera)
            # rx, ry = float(camera_pos_w[0]), float(camera_pos_w[1])
            # yaw_r  = yaw_from_quat_wxyz(camera_quat_w_wxyz)   # -> radians

            # # --- position error in WORLD
            # ex_w = gx - rx
            # ey_w = gy - ry

            # # --- rotate error into ROBOT frame: e_r = R_wr^T * e_w
            # R_wr = W_T_R[:3, :3].detach().cpu().numpy()    # robot->world
            # e_r  = R_wr.T @ np.array([ex_w, ey_w, 0.0], dtype=float)
            # ex_r, ey_r = float(e_r[0]), float(e_r[1])

            # # --- heading error (no rotation; wrap to [-pi, pi])
            # def wrap_to_pi(a): 
            #     return (a + np.pi) % (2*np.pi) - np.pi
            # yaw_err = wrap_to_pi(gth - yaw_r)

            # # --- simple P controller (base-frame command)
            # k_v, k_w = 1.0, 1.5     # gains
            # vmax, wmax = 0.8, 1.2   # limits
            # pos_tol, ang_tol = 0.15, 0.10

            # dist = float(np.hypot(ex_w, ey_w))
            # if dist < pos_tol and abs(yaw_err) < ang_tol:
            #     vx = vy = wz = 0.0
            # else:
            #     vx = float(np.clip(k_v * ex_r, -vmax, vmax))
            #     vy = 0.0  # or: float(np.clip(k_v * ey_r, -vmax, vmax)) if you want lateral motion
            #     wz = float(np.clip(k_w * yaw_err, -wmax, wmax))


        # Poll pygame events so joystick state updates
        if args_cli.teleop:
            print("[DEBUG] Using teleop mode")
            pygame.event.pump()
            term = env.unwrapped.command_manager.get_term("base_velocity")  # UniformVelocityCommand

            # scale joystick commands
            lin_x_scaling = 1.0
            lin_y_scaling = 1.0
            ang_w_scaling = 1.0
            if latest:
                vx = lin_x_scaling * -0.1 #error_in_robot_frame[0]
                # vx = lin_x_scaling * (-joystick.get_axis(1))   # forward/back (1.5)
                # vy = lin_y_scaling * ( joystick.get_axis(0))   # left/right (1.5)
                vy = lin_y_scaling * 0.1 #error_in_robot_frame[1]   # left/right (1.5)
                wz = ang_w_scaling * 0.1 #error_th_world
                # wz = ang_w_scaling * ( -joystick.get_axis(2))   # yaw rate (1.2)
            else:
                vx = lin_x_scaling * 0.0
                # vx = lin_x_scaling * (-joystick.get_axis(1))   # forward/back (1.5)
                # vy = lin_y_scaling * ( joystick.get_axis(0))   # left/right (1.5)
                vy = lin_y_scaling * 0   # left/right (1.5)
                wz = ang_w_scaling * 0


            # write directly into the term's tensor (shape = [N, 3])
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

    # save frame capture data
    frame_capture_path = os.path.join('/workspace/isaaclab/my_data', "frame_capture_data.npy")
    np.save(frame_capture_path, frame_capture_data)
    frame_capture_path = os.path.join('/workspace/isaaclab/my_data', "fixed_cam_frame_capture_data.npy")
    np.save(frame_capture_path, fixed_cam_frame_capture_data)
    print(f"[INFO] Saved frame capture data to: {frame_capture_path}")
    print(f"[INFO] Captured {len(frame_capture_data['frame_numbers'])} frames at intervals: {frame_capture_data['frame_numbers']}")

    # close the simulator
    env.close()
    # Close the RGBD publisher
    runner.stop()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
