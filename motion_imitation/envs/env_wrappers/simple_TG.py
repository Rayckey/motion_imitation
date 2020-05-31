# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple openloop trajectory generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
from gym import spaces
import numpy as np
from robots import laikago_pose_utils

class SimpleTG(object):
    """A trajectory generator that return constant motor angles."""

    def __init__(
            self, init_params, upstream_params, leg_id
    ):

        # We will need this I think
        # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        ##################################
        # tunable parameters
        # phi_leg, delta_phi, alpha_tg, Ae, Cs, t_p, theta, z, h_tg, k_sle, phi_t, f_tg, beta

        self._alpha_tg = 0
        self._Ae = 0
        self._Cs = 0
        self._theta = 0
        self._z = 0
        self._h_tg = 0
        self._k_sle = 0
        self._delta_phi = 0
        self._beta = 0

        self.unpack_params(init_params)

        # upstream tunable
        self._f_tg = upstream_params[0]

        # from upstream untunable
        self._phi_t = 0
        self._phi_leg = 0
        self._t_p = 0

        self._l_hip = 0.054
        self._l_upper = 0.25
        self._l_lower = 0.25



        # for matching the URDF direction
        self._gut_direction = 1

        self.set_leg_ID(leg_id)


        self._leg_id = leg_id

    def reset(self):
        pass

    def set_leg_ID(self, id):

        if id == 0 or id == 2:
            self._gut_direction = -1
        else:
            self._l_hip *= -1

    def _update_phi_leg(self, phi_t, phi_diff):
        phi_leg = np.mod(phi_t + phi_diff, 2.0 * np.pi)
        return phi_leg

    def _sync_phi_t(self, phi_t):
        self._phi_t = phi_t
        return

    def _compute_t_prime(self, phi_leg, beta):
        if 2 * np.pi * beta > phi_leg >= 0:
            t_p = phi_leg / (2.0 * beta)
        else:
            t_p = 2 * np.pi - (2 * np.pi - phi_leg) / (2 * (1 - beta))
        return t_p

    def _assemble_leg_height(self, phi_leg, beta, k_sle, t_p, Ae):
        if 2 * np.pi * beta > phi_leg >= 0:
            h_tg = self._h_tg
        else:
            h_tg = self._h_tg + (-k_sle * Ae * np.sin(t_p))
        return h_tg

    def _genertate_trajectory(self, alpha_tg, Ae, Cs, t_p, h_leg, theta, z):
        swing = Cs + alpha_tg * np.cos(t_p)

        # print('Swing')
        # print(np.cos(t_p))

        y = h_leg + Ae * np.sin(t_p) + theta * np.cos(t_p)
        x = - np.tan(swing) * y
        return x[0], y[0], z

    def unpack_params(self, params):
        self._alpha_tg = params[0]
        self._Ae = params[1]
        self._Cs = params[2]
        self._theta = params[3]
        self._z = params[4]
        self._h_tg = params[5]
        self._k_sle = params[6]
        self._delta_phi = params[7]
        self._beta = params[8]

    def get_IK(self, tar):
        x, y, z = tar

        l2 = np.sqrt(y * y + z * z)
        l1 = self._l_hip

        l_upper = self._l_upper
        l_lower = self._l_lower


        if abs(l1/l2) > 1:
            print(l1)
            print(l2)

        phi1 = np.arcsin(l1 / l2)
        phi2 = np.arctan2(-z, -y)

        # print(phi1)
        # print(phi2)
        theta = phi1 + phi2
        # print(theta)
        tip_2_hip_extent = -(y * np.cos(theta) + z * np.sin(theta))

        tip_2_hip_x = x

        # print(tip_2_hip_x)
        # print(tip_2_hip_extent)

        c = np.sqrt(x * x + tip_2_hip_extent * tip_2_hip_extent)

        inner_upper = np.arccos((c * c + l_upper * l_upper - l_lower * l_lower) / (2.0 * c * l_upper))

        tip_2_hip_swing = -np.arctan2(x, tip_2_hip_extent)

        inner_lower = np.arccos((l_upper * l_upper + l_lower * l_lower - c * c) / (2.0 * l_lower * l_upper))

        return np.array([theta, tip_2_hip_swing - inner_upper, np.pi - inner_lower])

    def get_trajectory(self, phi_t):
        self._sync_phi_t(phi_t=phi_t)

        self._phi_leg = self._update_phi_leg(phi_t=self._phi_t, phi_diff=self._delta_phi)

        self._t_p = self._compute_t_prime(phi_leg=self._phi_leg, beta=self._beta)
        h_leg = self._assemble_leg_height(phi_leg=self._phi_leg, beta=self._beta, k_sle=self._k_sle, t_p=self._t_p,
                                          Ae=self._Ae)

        # print('h_leg')
        # print(h_leg)

        tar = self._genertate_trajectory(alpha_tg=self._alpha_tg, Ae=self._Ae, Cs=self._Cs, t_p=self._t_p,
                                         h_leg=h_leg, theta=self._theta, z=self._z)

        # if self._leg_id == 0:
            # print("tar is ")
            # print(tar)

        # get that Ik in here
        res = self.get_IK(tar=tar)

        # if self._leg_id == 0:
            # print("from IK")
            # print(res)

        # account for motor direction
        # res[0] *= self._gut_direction
        res[0] *= 0
        res[1] *= -1
        res[2] *= -1

        # account for motor offset
        # res[1] -= laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE
        # res[2] -= laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE





        return res
