"""Simple  trajectory generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
from gym import spaces
import numpy as np

from robots import laikago_pose_utils

from envs.env_wrappers import simple_TG


class SimpleTGGroup(object):
    """A trajectory generator that return constant motor angles."""

    def __init__(
            self, init_lg_param=None,
            init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
            init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
            init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE,
            action_limit=0.2, is_touting=2, init_f_tg=1
    ):
        """Initializes the controller."""
        self._pose = np.array(
            attr.astuple(
                laikago_pose_utils.LaikagoPose(
                    abduction_angle_0=0,
                    hip_angle_0=0,
                    knee_angle_0=0,
                    abduction_angle_1=0,
                    hip_angle_1=0,
                    knee_angle_1=0,
                    abduction_angle_2=0,
                    hip_angle_2=0,
                    knee_angle_2=0,
                    abduction_angle_3=0,
                    hip_angle_3=0,
                    knee_angle_3=0)))

        action_high = np.array([action_limit] * 12)

        self._is_touting = is_touting

        # action_high = np.zeros([12])
        # action_high += action_limit

        # set the action bound
        if init_lg_param is None:
            print('Using default parameters for TG')
            init_lg_param = np.array([init_f_tg])
            for leg_num in range(0, 4):
                init_lg_param = np.concatenate([init_lg_param, self.get_default_params(leg_num)])
            print(init_lg_param)

        self._init_lg_param = init_lg_param

        lg_param_high = self.get_default_upper_bound(0)
        lg_param_low = self.get_default_lower_bound(0)
        for leg_num in range(1, 4):
            lg_param_high = np.concatenate([lg_param_high, self.get_default_upper_bound(leg_num)])
            lg_param_low = np.concatenate([lg_param_low, self.get_default_lower_bound(leg_num)])

        action_low = np.concatenate([-action_high, lg_param_low])
        action_high = np.concatenate([action_high, lg_param_high])

        # print(action_high)

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)

        # obs_space_upper = np.ones([4])*np.pi*2.0
        # self.observation_space = spaces.Box(-obs_space_upper, obs_space_upper, dtype=np.float32)

        # print("Action space shape is ")
        # print(self.action_space.shape)

        assert init_lg_param.size is 1 + 9 * 4

        self._time = 0
        self._phi_t = 0
        self._f_tg = self.unpack_params(init_lg_param)

        fr_tg = simple_TG.SimpleTG(init_params=self.unpack_params(init_lg_param, 0),
                                   upstream_params=self.unpack_params(init_lg_param), leg_id=0)

        fl_tg = simple_TG.SimpleTG(init_params=self.unpack_params(init_lg_param, 1),
                                   upstream_params=self.unpack_params(init_lg_param), leg_id=1)

        rr_tg = simple_TG.SimpleTG(init_params=self.unpack_params(init_lg_param, 2),
                                   upstream_params=self.unpack_params(init_lg_param), leg_id=2)

        rl_tg = simple_TG.SimpleTG(init_params=self.unpack_params(init_lg_param, 3),
                                   upstream_params=self.unpack_params(init_lg_param), leg_id=3)

        self._tg = [fr_tg, fl_tg, rr_tg, rl_tg]


        # ====== debug information
        self._counter = 0
        # ========================

    def reset(self):
        self._counter = 0
        pass

    def get_default_params(self, leg_id):

        gap = np.pi / 2.0

        # self._alpha_tg = params[0]
        # self._Ae = params[1]
        # self._Cs = params[2]
        # self._theta = params[3]
        # self._z = params[4]
        # self._h_tg = params[5]
        # self._k_sle = params[6]
        # self._delta_phi = params[7]
        # self._beta = params[8]

        res = np.array([np.pi / 10.0, 0.08, 0, 0, 0, -0.275, 0.8, 0, 0.3])

        # set touting phase change
        if self._is_touting == 0:
            res[7] = gap * leg_id
        else:
            if self._is_touting == 1:
                if leg_id == 0 or leg_id == 1:
                    res[7] = np.pi
                else:
                    res[7] = 0
            if self._is_touting == 2:
                if leg_id == 0 or leg_id == 3:
                    res[7] = 0
                else:
                    res[7] = np.pi

        # set hip offset
        if leg_id == 0 or leg_id == 2:
            res[4] = 0.054
        else:
            res[4] = -0.054

        # res[4] -= 0.05


        return res

    def get_default_upper_bound(self, leg_id):
        # f_tg = np.array([1.5])
        # indie = np.array([np.pi / 4.0, 0.05, np.pi / 4.0, 0.3, 0.1, 0.05, 0.5, np.pi / 4.0, 0.2])

        res = np.array([np.pi / 16.0, 0.02, 0.1, 0.1, 0.05, 0.025, 0.1, np.pi, 0.2])



        if self._is_touting == 0:
            res[7] = np.pi
        else:
            res[7] = np.pi / 4
            # res[8] = 0.2

        # set zeros, use this when testing
        # res = np.zeros(9)

        return res

    def get_default_lower_bound(self, leg_id):

        res = np.array([np.pi / 16.0, 0.01, 0.1, 0.1, 0.05, 0.025, 0.2, np.pi, 0.1])

        if self._is_touting == 0:
            res[7] = np.pi
        else:
            res[7] = np.pi/4
            # res[8] = 0.2

        res *= -1

        # set zeros, use this when testing
        # res = np.zeros(9)

        return res

    def _update_phi_t(self, f_tg, current_time=None):

        if current_time is None:
            phi_t = np.mod(self._phi_t + 2.0 * np.pi * self._f_tg, 2.0 * np.pi)
        else:
            phi_t = np.mod(current_time * f_tg * 2.0 * np.pi, 2.0 * np.pi)

        self._phi_t = phi_t
        return

    def get_action(self, current_time=None, input_action=None):
        """Computes the trajectory according to input time and action.
    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input [leg correction] and [trajectory parameters} from a NN controller.
    Returns:
      A numpy array. The desired motor angles.
    """

        # either get the current time to update or get time increment
        # probably the former
        self._time = current_time
        self._update_phi_t(self._f_tg, current_time=current_time)

        num_joint = 12
        num_joint_in_leg = 3

        tg_pose = np.zeros([num_joint])

        # retrieve from TG
        # print(self._f_tg)

        input_action = np.clip(input_action,self.action_space.low,self.action_space.high)

        input_param = np.concatenate([self._f_tg, input_action[num_joint:]]) + self._init_lg_param

        for leg_num in range(len(self._tg)):
            self._tg[leg_num].unpack_params(self.unpack_params(params=input_param, key=leg_num))

            # print(leg_num * num_joint_in_leg)
            #
            # print((leg_num + 1) * num_joint_in_leg)
            #
            # print(self._tg[leg_num].get_trajectory(self._phi_t))

            tg_pose[leg_num * num_joint_in_leg: (leg_num + 1) * num_joint_in_leg] = \
                self._tg[leg_num].get_trajectory(self._phi_t)

        # print("input is")
        # print(input_action[:num_joint])
        # print("output is")
        # print(tg_pose)

        return self._pose + input_action[:num_joint] + tg_pose
        # return self._pose

    def get_observation(self, input_observation):
        """Get the trajectory generator's observation."""

        phi_leg = np.zeros([4])
        for leg_num in range(0,4):
            phi_leg[leg_num] = self._tg[0].phi_leg

        return np.concatenate([input_observation, phi_leg])

        # return input_observation

    def unpack_params(self, params, key=-1):

        num_shared = 1
        num_indie = 9

        if key == -1:
            return params[:num_shared]
        elif key == 0:
            return params[num_shared:(num_shared + num_indie)]
        elif key == 1:
            return params[(num_shared + num_indie):(num_shared + num_indie * 2)]
        elif key == 2:
            return params[(num_shared + num_indie * 2):(num_shared + num_indie * 3)]
        elif key == 3:
            return params[(num_shared + num_indie * 3):]
