# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.diablo_vmc.diablo_vmc_config import (
    DiabloVMCCfg,
    DiabloVMCCfgPPO,
)


class DiabloVMCFlatCfg(DiabloVMCCfg):

    class terrain(DiabloVMCCfg.terrain):
        mesh_type = "plane"
    class rewards(DiabloVMCCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0

            base_height = 0.5
            nominal_state = -0.1
            lin_vel_z = -0.1e-3
            ang_vel_xy = -0.05
            orientation = -5.0

            dof_vel = -5e-5
            dof_acc = -2.5e-7
            torques = -0.1e-4
            action_rate = -0.001
            action_smooth = -0.001

            collision = -10.0
            dof_pos_limits = -0.1

            theta_limit = -0.01
            same_l = -0.1e-8
            # special for wheel
            # wheel_vel = -0.01

        base_height_target = 0.30
    class control(DiabloVMCCfg.control):
        l0_offset = 0.20
        feedforward_force = 60.0  # [N]

        # kp_theta = 60.0  # [N*m/rad]
        # kd_theta = 10.0  # [N*m*s/rad]
        # kp_l0 = 1000.0  # [N/m]
        # kd_l0 = 50.0  # [N*s/m]

        # real max
        kp_theta = 10.0  # [N*m/rad]
        kd_theta = 1.0  # [N*m*s/rad]
        kp_l0 = 300.0  # [N/m]
        kd_l0 = 8.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.8}  # [N*m*s/rad]

    class domain_rand(DiabloVMCCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-2.0, 3.0]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.2, 0.2]
        randomize_action_delay = True
        delay_ms_range = [0, 10]
class DiabloVMCFlatCfgPPO(DiabloVMCCfgPPO):

    class runner(DiabloVMCCfgPPO.runner):
        # logging
        policy_class_name = (
            "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence
        )
        # experiment_name = "diablo_vmc_flat_state_estimate"
        experiment_name = "diablo_vmc_flat"
        max_iterations = 30000
