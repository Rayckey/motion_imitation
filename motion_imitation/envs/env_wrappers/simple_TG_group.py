"""Simple  trajectory generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
from gym import spaces
import numpy as np

from robots import laikago_pose_utils

from simple_TG import SimpleTG


class SimpleTGGroup(object):
    """A trajectory generator that return constant motor angles."""

    def __init__(
            self, init_lg_param=None,
            init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
            init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
            init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE,
            action_limit=0.5,
    ):
        """Initializes the controller."""
        self._pose = np.array(
            attr.astuple(
                laikago_pose_utils.LaikagoPose(
                    abduction_angle_0=init_abduction,
                    hip_angle_0=init_hip,
                    knee_angle_0=init_knee,
                    abduction_angle_1=init_abduction,
                    hip_angle_1=init_hip,
                    knee_angle_1=init_knee,
                    abduction_angle_2=init_abduction,
                    hip_angle_2=init_hip,
                    knee_angle_2=init_knee,
                    abduction_angle_3=init_abduction,
                    hip_angle_3=init_hip,
                    knee_angle_3=init_knee)))

        action_high = np.array([action_limit] * 12)

        # set the action bound
        if init_lg_param is None:
            print('Using default parameters for TG')
            init_lg_param = self.get_default_params()

        lg_param_high = init_lg_param + self.get_default_bound()
        lg_param_low = init_lg_param - self.get_default_bound()

        action_low = np.concatenate([-action_high, lg_param_low])
        action_high = np.concatenate([action_high, lg_param_high])

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)

        assert init_lg_param.size is 1 + 9 * 4

        self._time = 0
        self._phi_t = 0
        self._f_tg = self.unpack_params(init_lg_param)

        fl_tg = SimpleTG(init_params=self.unpack_params(init_lg_param, 0),
                         upstream_params=self.unpack_params(init_lg_param), leg_id=0)

        fr_tg = SimpleTG(init_params=self.unpack_params(init_lg_param, 1),
                         upstream_params=self.unpack_params(init_lg_param), leg_id=1)

        rl_tg = SimpleTG(init_params=self.unpack_params(init_lg_param, 2),
                         upstream_params=self.unpack_params(init_lg_param), leg_id=2)

        rr_tg = SimpleTG(init_params=self.unpack_params(init_lg_param, 3),
                         upstream_params=self.unpack_params(init_lg_param), leg_id=3)

        self._tg = [fl_tg, fr_tg, rl_tg, rr_tg]

    def reset(self):
        pass

    def get_default_params(self):
        f_tg = 2
        gap = np.pi / 2.0
        indie = np.array([np.pi / 4, 0.07, 0, 0, 0, -0.35, 0.5, 0, 0.3])
        res = np.concatenate([f_tg, indie])
        for leg_num in range(1, 4):
            indie[7] += gap
            res = np.concatenate([res, indie])

        return res

    def get_default_bound(self):
        f_tg = 1.5
        indie = np.array([np.pi / 4.0, 0.05, np.pi / 4.0, 0.3, 0.1, 0.1, 0.5, np.pi / 4.0, 0.2])
        res = np.concatenate([f_tg, indie])
        for leg_num in range(1, 4):
            res = np.concatenate([res, indie])

        return res

    def _update_phi_t(self, f_tg, current_time=None):

        if current_time is None:
            phi_t = np.mod(self._phi_t + 2.0 * np.pi * self._f_tg, 2.0 * np.pi)
        else:
            phi_t = current_time * f_tg

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

        tg_pose = np.array([num_joint])

        # retrieve from TG
        input_param = input_action[num_joint:]
        for leg_num in range(len(self._tg)):
            self._tg[leg_num].unpack_params(self.unpack_params(params=input_param, key=leg_num))
            tg_pose[leg_num * num_joint_in_leg: (leg_num + 1) * num_joint_in_leg] = \
                self._tg[leg_num].get_trajectory(self._phi_t)

        return self._pose + input_action[:num_joint] + tg_pose

    def get_observation(self, input_observation):
        """Get the trajectory generator's observation."""

        return np.concatenate(input_observation, self._phi_t)

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
