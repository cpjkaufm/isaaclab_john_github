from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from .forklift_c import FORKLIFT_C_CFG
from isaaclab.markers import VisualizationMarkers

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.assets import RigidObject

import math

# Set circle_testing to true to have the robot go spinny spinny
circle_testing = True

@configclass
class ForkliftCircleEnvCfg(DirectRLEnvCfg):
    decimation = 4
    # Reflects the length of training segment
    episode_length_s = 40.0

    if circle_testing or 1==1: # The state space used to be dynamic based on what test was being done, but now it's static
        action_space = 2
        observation_space = 5

    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 30, render_interval=decimation)
    robot_cfg: ArticulationCfg = FORKLIFT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Register the joints to their respective functions
    throttle_dof_name = [
        "left_front_wheel_joint",
        "left_back_wheel_joint",
        "right_front_wheel_joint",
        "right_back_wheel_joint",
    ]
    steering_dof_name = [
        "left_rotator_joint",
        "right_rotator_joint",
    ]

    # Some parameters for scene generation
    env_spacing = 60.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class ForkliftCircleEnv(DirectRLEnv):
    cfg: ForkliftCircleEnvCfg

    def __init__(self, cfg: ForkliftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._throttle_dof_idx, _ = self.forklift_c.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.forklift_c.find_joints(self.cfg.steering_dof_name)

        # Populate the states of the joints with 0 values to start
        # The num in (self.num_envs, _) must reflect the number of joints associated with that function
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)

        self.env_spacing = self.cfg.env_spacing
        self.course_width_coefficient = 2.0

    def _setup_scene(self):
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )


        # Setup rest of the scene
        self.forklift_c = Articulation(self.cfg.robot_cfg)
        self.object_state = []
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["forklift_c"] = self.forklift_c

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 3.0

        """ This code had to be changed to reflect the 4 throttle joints and 2 steering joint """
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * (throttle_scale)
        self._throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max)
        self._throttle_state = self._throttle_action
        
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action


    def _apply_action(self) -> None:
        self.forklift_c.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.forklift_c.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)


    def _get_observations(self) -> dict:

        # Defines the input that we give to the ML algorithm
        obs_parts = [
            self.forklift_c.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
            self.forklift_c.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
            self.forklift_c.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
            self._throttle_state[:, 0].unsqueeze(dim=1),
            self._steering_state[:, 0].unsqueeze(dim=1),
        ]

        obs = torch.cat(obs_parts, dim=-1)
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    

    #####################################
    #
    #   This is THE reward function
    #
    #####################################
    def _get_rewards(self) -> torch.Tensor:

        # Reverse reward logic
        fwd_dir = self.forklift_c.data.root_lin_vel_w[..., :2]  # Approximate forward direction
        fwd_dir = torch.nn.functional.normalize(fwd_dir, dim=-1)

            
        # This results in the truck driving in a sharp circle, albeit slowly
        steer_joint_positions = self.forklift_c.data.joint_pos[:, self._steering_dof_idx]
        steer_penalty = torch.sum(torch.abs(steer_joint_positions), dim=1) # Penalize large steering angles 
        composite_reward = steer_penalty
        
        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        print("Final reward was: ", composite_reward)

        return composite_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get_dones is expected by IsaacLab.
        I think it is used for saying whether an environment should be reset or not?
        If either return is true (it needs two returns), then it does something.
        """
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, task_failed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.forklift_c._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        default_state = self.forklift_c.data.default_root_state[env_ids]
        forklift_c_pose = default_state[:, :7]
        forklift_c_velocities = default_state[:, 7:]
        joint_positions = self.forklift_c.data.default_joint_pos[env_ids]
        joint_velocities = self.forklift_c.data.default_joint_vel[env_ids]

        forklift_c_pose[:, :3] += self.scene.env_origins[env_ids]
        forklift_c_pose[:, 0] -= self.env_spacing / 2
        forklift_c_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        forklift_c_pose[:, 3] = torch.cos(angles * 0.5)
        forklift_c_pose[:, 6] = torch.sin(angles * 0.5)

        self.forklift_c.write_root_pose_to_sim(forklift_c_pose, env_ids)
        self.forklift_c.write_root_velocity_to_sim(forklift_c_velocities, env_ids)
        self.forklift_c.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)