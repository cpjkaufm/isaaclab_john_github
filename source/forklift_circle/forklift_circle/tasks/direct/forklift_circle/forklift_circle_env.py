from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.imu import Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from .forklift_c import FORKLIFT_C_CFG
from .forklift_c import forklift_throttle_dof_name
from .forklift_c import forklift_steering_dof_name

from .hybot_c import HYBOT_C_CFG
from .hybot_c import hybot_throttle_dof_name
from .hybot_c import hybot_steering_dof_name

ROBOT_TYPE_FORKLIFT = 0
ROBOT_TYPE_HYBOT = 1

#Change this to switch between forklift and hybot
ROBOT_TYPE = ROBOT_TYPE_HYBOT

@configclass
class ForkliftCircleEnvCfg(DirectRLEnvCfg):
    decimation = 4
    
    # Control how long an episode lasts
    episode_length_s = 30.0

    if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
        action_space = 2
        observation_space = 5

    elif ROBOT_TYPE == ROBOT_TYPE_HYBOT:
        action_space = 1
        observation_space = 1

    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 30, render_interval=decimation)

    if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
        robot_cfg: ArticulationCfg = FORKLIFT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        throttle_dof_name = forklift_throttle_dof_name
        steering_dof_name = forklift_steering_dof_name
    elif ROBOT_TYPE == ROBOT_TYPE_HYBOT:
        robot_cfg: ArticulationCfg = HYBOT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        throttle_dof_name = hybot_throttle_dof_name
        steering_dof_name = hybot_steering_dof_name

        #imu_robot_base: ImuCfg = HYBOT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot/base_imu")

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
        #self._imu_stuff = self.cfg.imu_robot_base

        # Populate the states of the joints with 0 values to start
        # The num in (self.num_envs, _) must reflect the number of joints associated with that function
        self._throttle_state = torch.zeros((self.num_envs, self.cfg.num_throttle_joints), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs, self.cfg.num_steer_joints), device=self.device, dtype=torch.float32)

        self.env_spacing = self.cfg.env_spacing
        self.course_width_coefficient = 0.0

        # Have it stop moving after n seconds
        self._timers = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._timers += 100.0


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

        #print("Linear Acceleration:", self._imu_stuff.data.lin_acc_b)
        #print("Angular Velocity:", self._imu_stuff.data.ang_vel_b)

        # The actions inputs comes from the RL algorithm, and is expected to be a tensor of shape (num_envs, action_space)
        # Action_space is the number of actions that the robot can take
        # The values of action_space start as two randomly generated numbers
        # But they will begin to get weighted towards rewarded values
        # :0 is throttle, :1 is steering

        if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
            throttle_scale = 50 # Previously 10.0
            throttle_max = 100 # Previously 50
            throttle_min = -throttle_max
        
        elif ROBOT_TYPE == ROBOT_TYPE_HYBOT:
            throttle_scale = 5.0
            throttle_max = 25.0
            throttle_min = -throttle_max


        # Use 4 for repeat_interlave and reshape to match the number of throttle joints
        self._throttle_action = actions[:, 0].repeat_interleave(self.cfg.num_throttle_joints).reshape((-1, self.cfg.num_throttle_joints)) * (throttle_scale)
        self._throttle_action = torch.clamp(self._throttle_action, throttle_min, throttle_max)
        self._throttle_state = self._throttle_action

        #print("Throttle action: ", self._throttle_action)
        
        if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
            # Pro-tip rapid steer angles change cause the truck to turn into a bucking bronco
            steering_scale = 0.2 # Previously 0.1
            steering_max = 10.0 # Previously 3.0
            steering_min = -steering_max

        elif ROBOT_TYPE == ROBOT_TYPE_HYBOT:
            steering_scale = 0.0
            steering_max = 0.0
            steering_min = -0.0

        if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
            # Use 2 for repeat_interleave and reshape to match the number of steering joints
            self._steering_action = actions[:, 1].repeat_interleave(self.cfg.num_steer_joints).reshape((-1, self.cfg.num_steer_joints)) * steering_scale
            self._steering_action = torch.clamp(self._steering_action, steering_min, steering_max)
            self._steering_state = self._steering_action

        #print("Steering action: ", self._steering_action)

        #print(self._steering_dof_idx)
        #print(self.forklift_c.data.joint_names)

    def _apply_action(self) -> None:
        """
        Apply the actions to the robot
        """
        self.forklift_c.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        
        if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
            self.forklift_c.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)


    def _get_observations(self) -> dict:

        # Defines the input that we give to the ML algorithm

        if ROBOT_TYPE == ROBOT_TYPE_FORKLIFT:
            obs_parts = [
                self.forklift_c.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.forklift_c.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.forklift_c.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ]

        elif ROBOT_TYPE == ROBOT_TYPE_HYBOT:
            obs_parts = [
                self._timers.unsqueeze(dim=1),
            ]


        obs = torch.cat(obs_parts, dim=-1)
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    

    def _get_rewards(self) -> torch.Tensor:
        """
        Based on the current state of the robot(s), calculate a reward function
        """

        #print("Root lin vel b: ", self.forklift_c.data.root_lin_vel_b)
        #print("Root ang vel w: ", self.forklift_c.data.root_ang_vel_w)

        ### Reward for throttle joint velocities
        throttle_joint_velocities = self.forklift_c.data.joint_vel[:, self._throttle_dof_idx]
        throttle_joint_positions = self.forklift_c.data.joint_pos[:, self._throttle_dof_idx]

        #print("Throttle joint velocities: ", throttle_joint_velocities)
        #print("Throttle joint positions: ", throttle_joint_positions)

        throttle_penalty = torch.sum(throttle_joint_velocities, dim=1)
        composite_reward = throttle_penalty
       

        ### Reward for steer angle joint positions
        steer_joint_velocities = self.forklift_c.data.joint_vel[:, self._steering_dof_idx]
        steer_joint_positions = self.forklift_c.data.joint_pos[:, self._steering_dof_idx]

        #print("Steer joint velocities: ", steer_joint_velocities)
        #print("Steer joint positions: ", steer_joint_positions)

        # Punish steer angles that aren't straight
        steer_penalty = 2.0 * torch.sum(torch.abs(steer_joint_positions), dim=1) 
        composite_reward -= steer_penalty
        
        ### Reward for stopping after n seconds
        
        # Decrement timers
        self._timers -= 1
        timer_passed = self._timers < 0
        #print("Timer passed: ", timer_passed)

        # Determine if the truck is stopped, and throttle is near 0
        lin_speed = torch.norm(self.forklift_c.data.root_lin_vel_b[:, :2], dim=-1)
        throttle = self.actions[..., 0]
        fully_stopped = (lin_speed < 0.1) & (torch.abs(throttle) < 0.05)

        # Reward for coming to a stop
        stopped_reward = 5.0 * timer_passed.float() * fully_stopped.float()
        composite_reward += stopped_reward

        # Punish for not being stopped
        abs_throttle_penalty = torch.sum(torch.abs(throttle_joint_velocities), dim=1)
        not_stopped_punish = -5.0 * timer_passed.float() * abs_throttle_penalty
        composite_reward += not_stopped_punish

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        #print("Final reward was: ", composite_reward)

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

        self._timers[env_ids] = 0.0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get_dones is expected by IsaacLab.
        I think it is used for saying whether an environment should be reset or not?
        If either return is true (it needs two returns), then it does something.
        """
        task_failed = self.episode_length_buf > self.max_episode_length

        #if task_failed.any():
        #    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return task_failed, task_failed