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
from .hybot_c import HYBOT_C_CFG
from .hybot_c import hybot_throttle_dof_name
from .hybot_c import hybot_steering_dof_name

ROBOT_TYPE = 1 # 0 for forklift with steering, 1 for straight line hybot

@configclass
class ForkliftCircleEnvCfg(DirectRLEnvCfg):
    decimation = 4
    
    # Control how long an episode lasts
    episode_length_s = 15.0

    if ROBOT_TYPE == 0:
        action_space = 2
        observation_space = 5

    elif ROBOT_TYPE == 1:
        action_space = 1
        observation_space = 4

    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 30, render_interval=decimation)

    if ROBOT_TYPE == 0:
        robot_cfg: ArticulationCfg = FORKLIFT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        throttle_dof_name = FORKLIFT_C_CFG.throttle_dof_name
        steering_dof_name = FORKLIFT_C_CFG.steering_dof_name
    elif ROBOT_TYPE == 1:
        robot_cfg: ArticulationCfg = HYBOT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        throttle_dof_name = hybot_throttle_dof_name
        steering_dof_name = []

    num_throttle_joints = len(throttle_dof_name)
    num_steer_joints = len(steering_dof_name)

    # Some parameters for scene generation
    env_spacing = 10.0 # Contols the initial spacing between envs at each reset
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class ForkliftCircleEnv(DirectRLEnv):
    cfg: ForkliftCircleEnvCfg

    def __init__(self, cfg: ForkliftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get references to the joints for throttle and steering
        self._throttle_dof_idx, _ = self.forklift_c.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.forklift_c.find_joints(self.cfg.steering_dof_name)

        # Populate the states of the joints with 0 values to start
        # The num in (self.num_envs, _) must reflect the number of joints associated with that function
        self._throttle_state = torch.zeros((self.num_envs, self.cfg.num_throttle_joints), device=self.device, dtype=torch.float32)

        if  ROBOT_TYPE == 0:
            self._steering_state = torch.zeros((self.num_envs, self.cfg.num_steer_joints), device=self.device, dtype=torch.float32)

        self.env_spacing = self.cfg.env_spacing
        self.course_width_coefficient = 0.0

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
        """
        Calculate values for the next step of the physics simulation
        """

        # The actions inputs comes from the RL algorithm, and is expected to be a tensor of shape (num_envs, action_space)
        # Action_space is the number of actions that the robot can take
        # The values of action_space start as two randomly generated numbers
        # But they will begin to get weighted towards rewarded values
        # :0 is throttle, :1 is steering

        if ROBOT_TYPE == 0:
            throttle_scale = 50 # Previously 10.0
            throttle_max = 100 # Previously 50
            throttle_min = -throttle_max
        
        elif ROBOT_TYPE == 1:
            throttle_scale = 10
            throttle_max = 25
            throttle_min = -throttle_max


        # Use 4 for repeat_interlave and reshape to match the number of throttle joints
        self._throttle_action = actions[:, 0].repeat_interleave(self.cfg.num_throttle_joints).reshape((-1, self.cfg.num_throttle_joints)) * (throttle_scale)
        self._throttle_action = torch.clamp(self._throttle_action, throttle_min, throttle_max)
        self._throttle_state = self._throttle_action

        print("Throttle action: ", self._throttle_action)
        
        if ROBOT_TYPE == 0:
            # Pro-tip rapid steer angles change cause the truck to turn into a bucking bronco
            steering_scale = 0.2 # Previously 0.1
            steering_max = 0.0 # Previously 3.0
            steering_min = -steering_max

            # Use 2 for repeat_interleave and reshape to match the number of steering joints
            self._steering_action = actions[:, 1].repeat_interleave(self.cfg.num_steer_joints).reshape((-1, self.cfg.num_steer_joints)) * steering_scale
            self._steering_action = torch.clamp(self._steering_action, steering_min, steering_max)
            self._steering_state = self._steering_action


    def _apply_action(self) -> None:
        """
        Apply the actions to the robot
        """
        self.forklift_c.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)

        if ROBOT_TYPE == 0:
            self.forklift_c.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)


    def _get_observations(self) -> dict:

        # Defines the input that we give to the ML algorithm

        if ROBOT_TYPE == 0:
            obs_parts = [
                self.forklift_c.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.forklift_c.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.forklift_c.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ]

        elif ROBOT_TYPE == 1:
            obs_parts = [
                self.forklift_c.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.forklift_c.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.forklift_c.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
            ]

        obs = torch.cat(obs_parts, dim=-1)
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    

    def _get_rewards(self) -> torch.Tensor:
        """
        Based on the current state of the robot(s), calculate a reward function
        """

        # Reward for throttle joint velocities
        throttle_joint_velocities = self.forklift_c.data.joint_vel[:, self._throttle_dof_idx]
        throttle_penalty = torch.sum(throttle_joint_velocities, dim=1)

        print("Throttle joint velocities: ", throttle_joint_velocities)

        composite_reward = throttle_penalty

        steer_joint_positions = self.forklift_c.data.joint_pos[:, self._steering_dof_idx]
        print("Steer joint positions: ", steer_joint_positions)
        

        # Reward for steer angle joint positions
        if ROBOT_TYPE == 0:
            steer_joint_positions = self.forklift_c.data.joint_pos[:, self._steering_dof_idx]
            steer_penalty = torch.sum(torch.abs(steer_joint_positions), dim=1) 
        
            composite_reward += steer_penalty
        
        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        print("Final reward was: ", composite_reward)

        return composite_reward


    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset the forklift with some pseudo-random init conditions
        """
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

        # At reset, set the forklift to a random angle in the range of 0 to angle_range radians
        angle_range = 0 * (torch.pi / 0.5)
        angles = angle_range * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        forklift_c_pose[:, 3] = torch.cos(angles * 0.5)
        forklift_c_pose[:, 6] = torch.sin(angles * 0.5)

        self.forklift_c.write_root_pose_to_sim(forklift_c_pose, env_ids)
        self.forklift_c.write_root_velocity_to_sim(forklift_c_velocities, env_ids)
        self.forklift_c.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get_dones is expected by IsaacLab.
        I think it is used for saying whether an environment should be reset or not?
        If either return is true (it needs two returns), then it does something.
        """
        task_failed = self.episode_length_buf > self.max_episode_length

        if task_failed.any():
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return task_failed, task_failed