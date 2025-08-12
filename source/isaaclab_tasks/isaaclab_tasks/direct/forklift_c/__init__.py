# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Forklift driving environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

""" Altered to reflect the name of the project: Isaac-Forklift-C-Direct-v0 """
gym.register(
    id="Isaac-Forklift-C-Direct-v0",
    entry_point=f"{__name__}.forklift_c_env:ForkliftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forklift_c_env:ForkliftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
