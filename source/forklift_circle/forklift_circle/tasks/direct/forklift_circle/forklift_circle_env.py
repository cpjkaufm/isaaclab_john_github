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
from .waypoint import WAYPOINT_CFG
from .forklift_c import FORKLIFT_C_CFG
from isaaclab.markers import VisualizationMarkers

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.assets import RigidObject

import math

# Two projects, currently working with STOP_TESTING
lift_testing = False
stop_testing = False
circle_testing = True

STOP_SIGN_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/StopSign",
    spawn=UsdFileCfg(
        # THIS PATH WILL NEED TO BE ALTERED FOR YOUR OWN CODE
        usd_path="/home/hybot/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_c/custom_assets/STOP.usd",
        scale=(0.2, 0.2, 0.2),
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0
        )
    )
)

@configclass
class ForkliftCircleEnvCfg(DirectRLEnvCfg):
    decimation = 4
    # Reflects the length of training segment
    episode_length_s = 40.0

    # The action_space is the num of outputs of the ML algorithm, the observation_space is the num of inputs
    if lift_testing:
        action_space = 3          # includes the output for forks
        observation_space = 10    # includes the pos & vel of the forks
    elif stop_testing:
        action_space = 2
        observation_space = 10
    else:
        action_space = 2
        observation_space = 8

    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 30, render_interval=decimation)
    robot_cfg: ArticulationCfg = FORKLIFT_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG

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
    if lift_testing:
        lift_dof_name = [
            "lift_joint",
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
        if lift_testing:
            self._lift_state = torch.zeros((self.num_envs,1), device=self.device, dtype=torch.float32)
            self._lift_dof_idx, _ = self.forklift_c.find_joints(self.cfg.lift_dof_name)

        # Other parameters for the waypoints and rewards
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._num_goals = 5
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.env_spacing = self.cfg.env_spacing

        # Random variables (from Leatherback)
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0
        self.position_tolerance = 0.8
        self.goal_reached_bonus = 10.0
        self.position_progress_weight = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight = 0.05
        self.stop_zone = 4.6    # Sensitive number, changes behavior of forklift

        ''' GLOBAL VARIABLES FOR STOP SIGN IMPLEMENTATION '''
        if stop_testing:
            self._stopping_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self._lingering_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self._has_stopped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._has_lingered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

        # Setup stop signs (one for each forklift)
        if stop_testing:
            self._stop_sign_positions = torch.zeros((self.num_envs, 2), device=self.device)
            self.stop_signs = RigidObject(STOP_SIGN_CFG)
            self.scene.rigid_objects["stop_signs"] = self.stop_signs

        # Setup rest of the scene
        self.forklift_c = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
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

        if lift_testing:
            lift_scale = 1.0
            lift_max = 0.8

            self._lift_action = actions[:, 2].repeat_interleave(1).reshape((-1, 1)) * lift_scale
            self._lift_action = torch.clamp(self._lift_action, -lift_max, lift_max)
            self._lift_state = self._lift_action

    def _apply_action(self) -> None:
        self.forklift_c.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.forklift_c.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)
        if lift_testing:
            self.forklift_c.set_joint_position_target(self._lift_state, joint_ids=self._lift_dof_idx)

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[self.forklift_c._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.forklift_c.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.forklift_c.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.forklift_c._ALL_INDICES, self._target_index, 1] - self.forklift_c.data.root_link_pos_w[:, 1],
            self._target_positions[self.forklift_c._ALL_INDICES, self._target_index, 0] - self.forklift_c.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        if lift_testing:
            lift_joint_pos = self.forklift_c.data.joint_pos[:, self._lift_dof_idx].squeeze(-1)
            lift_joint_vel = self.forklift_c.data.joint_vel[:, self._lift_dof_idx].squeeze(-1)

        if stop_testing:
            dist_to_stop = torch.norm(self._stop_sign_positions - self.forklift_c.data.root_pos_w[:, :2], dim=-1, keepdim=True)
            is_close_to_stop = (dist_to_stop < self.stop_zone).float()


        # Defines the input that we give to the ML algorithm
        obs_parts = [
            self._position_error.unsqueeze(dim=1),
            torch.cos(self.target_heading_error).unsqueeze(dim=1),
            torch.sin(self.target_heading_error).unsqueeze(dim=1),
            self.forklift_c.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
            self.forklift_c.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
            self.forklift_c.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
            self._throttle_state[:, 0].unsqueeze(dim=1),
            self._steering_state[:, 0].unsqueeze(dim=1),
        ]

        if lift_testing:
            obs_parts += [
                lift_joint_pos.unsqueeze(dim=1),
                lift_joint_vel.unsqueeze(dim=1),
            ]

        # Additional observations for STOP_TESTING (10 total)
        if stop_testing:
            obs_parts += [
                dist_to_stop,
                is_close_to_stop,
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
        position_progress_rew = self._previous_position_error - self._position_error
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals -1)
        self._target_index = self._target_index % self._num_goals

        # Reverse reward logic
        fwd_dir = self.forklift_c.data.root_lin_vel_w[..., :2]  # Approximate forward direction
        fwd_dir = torch.nn.functional.normalize(fwd_dir, dim=-1)

        # From Leatherback
        env_ids = self.forklift_c._ALL_INDICES
        goal_pos = self._target_positions[env_ids, self._target_index]  # [num_envs, 2]
        to_wp = goal_pos - self.forklift_c.data.root_pos_w[..., :2]
        to_wp = torch.nn.functional.normalize(to_wp, dim=-1)

        facing_dot = torch.sum(fwd_dir * to_wp, dim=-1)
        reverse_condition = (facing_dot < -0.5) & (self.actions[..., 0] < 0)
        reverse_bonus = reverse_condition.float() * 1.5

        forward_wrong_way = (facing_dot < -0.5) & (self.actions[..., 0] > 0)
        forward_penalty = forward_wrong_way.float() * -1.5

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            goal_reached * self.goal_reached_bonus +
            reverse_bonus + forward_penalty
        )

        #print("Initial Reward: ", composite_reward)


        # Reward information for moving the forks (old project)
        if lift_testing:
            # Distance to goal (already computed)
            max_lift_distance = 2.0
            normalized_dist = torch.clamp(self._position_error / max_lift_distance, 0.0, 1.0)

            # Closer = higher lift
            desired_lift_height = 1.0 - normalized_dist

            # Get current lift position
            lift_joint_pos = self.forklift_c.data.joint_pos[:, self._lift_dof_idx].squeeze(-1)

            # Reward closeness to target height
            lift_error = torch.abs(lift_joint_pos - desired_lift_height)
            lift_alignment_reward = torch.exp(-lift_error / 0.1)  # sharper = tighter tolerance

            composite_reward += lift_alignment_reward * 1.0


        ''' REWARD SECTION FOR STOPPING AT A STOP SIGN '''
        if stop_testing:
            # Determines if the forklift is inside of the stop zone (in a circle around stop sign)
            forklift_pos = self.forklift_c.data.root_pos_w[:, :2]
            dist_to_stop = torch.norm(self._stop_sign_positions - forklift_pos, dim=-1)
            in_stop_zone = dist_to_stop < self.stop_zone

            lin_speed = torch.norm(self.forklift_c.data.root_lin_vel_b[:, :2], dim=-1)
            throttle = self.actions[..., 0]

            # Boolean if the forklift has stopped moving
            fully_stopped = (lin_speed < 0.1) & (torch.abs(throttle) < 0.05)

            # Timers determine how long the forklift has been in the stop zone
            still_in_zone = in_stop_zone & fully_stopped
            self._stopping_timer[still_in_zone] += 1
            self._stopping_timer[~still_in_zone] = 0

            self._lingering_timer[still_in_zone] += 1
            self._lingering_timer[~still_in_zone] = 0

            # Forklift is supposed to stay in the zone from 4 to 6 seconds
            self._has_stopped = self._stopping_timer >= 4
            self._has_lingered = self._lingering_timer >= 6

            # If there was a full stop, reward
            # If lingered past 6 seconds, punish heavily (doesn't seem to work)
            full_stop_complete = (self._stopping_timer == 4).float() * 5.0
            lingering_in_stop = (self._has_lingered == 6).float() * -50.0
        
            # If left without reaching 4 seconds, punish
            left_early = in_stop_zone & ~self._has_stopped & (torch.abs(throttle) > 0.05)
            penalty_for_ealy_move = left_early.float() * -5.0


            # Add rewards
            composite_reward += full_stop_complete + penalty_for_ealy_move + lingering_in_stop

        ''' END OF REWARD SECTION FOR STOPPING AT A STOP SIGN '''

        if circle_testing:
            
            steer_joint_positions = self.forklift_c.data.joint_pos[:, self._steering_dof_idx].squeeze(-1)
            first_steer_pos = steer_joint_positions[0][0].item()
            second_steer_pos = steer_joint_positions[0][1].item()
            
            print("Steer joint positions are: ", steer_joint_positions)
            print("First is: ", first_steer_pos)
            print("Second is: ", second_steer_pos)

            # We only go one way on this rig
            turn_penalty = 0.0
            if first_steer_pos < -0.2:
                turn_penalty = -20.0
            elif first_steer_pos > 0.2:
                turn_penalty = 20.0

            tensor_reward = torch.tensor(first_steer_pos, device=self.device, dtype=torch.float32)
            composite_reward = tensor_reward
        

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        print("Final reward was: ", composite_reward)

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

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

        # target_positions and marker_pos are the positions of the waypoints
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        width_variance = 10.0 # previously 10
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) * width_variance + self.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        ''' STOP SIGN ORIENTATION AND DISPLAY BELOW '''

        if stop_testing:
            angle_rad = math.radians(90)
            cos = math.cos(angle_rad / 2)
            sin = math.sin(angle_rad / 2)

            upright_quat = torch.tensor([sin, 0.0, 0.0, cos], device=self.device)

            num_reset = len(env_ids)
            random_stop_indices = torch.randint(0, self._num_goals, (num_reset,), device=self.device)
            offset = torch.tensor([0.0, 2.5], device=self.device)
            self._stop_sign_positions[env_ids] = self._target_positions[env_ids, random_stop_indices] + offset

            stop_pose = torch.zeros((num_reset, 7), device=self.device)
            stop_pose[:, :2] = self._stop_sign_positions[env_ids]       # (X, Y)
            stop_pose[:, 2] = 0.0                                       # height (z)
            stop_pose[:, 3:] = upright_quat                             # w of quaternion

            self.stop_signs.write_root_pose_to_sim(stop_pose, env_ids)

        ''' END OF STOP SIGN ORIENTATION AND DISPLAY '''

        current_target_positions = self._target_positions[self.forklift_c._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.forklift_c.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.forklift_c.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.forklift_c.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.forklift_c.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()