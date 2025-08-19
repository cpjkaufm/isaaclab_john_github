# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the forklift robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

# USD path with proper resolution for cross-platform compatibility
USD_PATH = os.path.join(WORKSPACE_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "direct", "forklift_john_c", "custom_assets", "forklift_john_c.usd")

FORKLIFT_JOHN_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    # Updated to reflect forklift_b.usd joints
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "left_front_wheel_joint": 0.0,
            "left_back_wheel_joint": 0.0,
            "right_front_wheel_joint": 0.0,
            "right_back_wheel_joint": 0.0,
            "left_rotator_joint": 0.0,
            "right_rotator_joint": 0.0,
            "lift_joint": 0.0,
        },
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit=100.0, #400.0,
            velocity_limit=50.0, #100.0,
            stiffness=0.0, #2000.0,
            damping=100000.0, #500.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*_rotator_joint"],
            effort_limit=2000.0, #4000.0,
            velocity_limit=100.0,
            stiffness=10000.0, #2000.0,
            damping=0.0, #300.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            effort_limit=2000.0,
            velocity_limit=5.0,
            stiffness=1000.0,
            damping=200.0,
        )
    },
)
