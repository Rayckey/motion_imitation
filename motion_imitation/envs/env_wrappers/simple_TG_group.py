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

from simple_TG import SimpleTG

class SimpleTGGroup(object):
  """A trajectory generator that return constant motor angles."""

  def __init__(
      self, init_lg_param,
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
    # action_high = np.array([action_limit] * 12)
    # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    assert init_lg_param.size is 1+9*4


    self._time = 0
    self._phi_t = 0
    self._f_tg = self.unpack_params(init_lg_param)


    fl_tg = SimpleTG(init_params = self.unpack_params(init_lg_param , 0) , upstream_params= self.unpack_params(init_lg_param))

    fr_tg = SimpleTG(init_params = self.unpack_params(init_lg_param , 1) , upstream_params= self.unpack_params(init_lg_param))

    rl_tg = SimpleTG(init_params = self.unpack_params(init_lg_param , 2) , upstream_params= self.unpack_params(init_lg_param))

    rr_tg = SimpleTG(init_params = self.unpack_params(init_lg_param , 3) , upstream_params= self.unpack_params(init_lg_param))

    self._tg = [fl_tg , fr_tg , rl_tg , rr_tg]

  def reset(self):
    pass

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
      input_action: A numpy array. The input [leg pose]  and [trajectory parameters} from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """

    # either get the current time to update or get time increment
    # probably the former
    self._time = current_time
    self._update_phi_t(self._f_tg , current_time = current_time)

    num_joint = 12
    num_joint_in_leg = 3

    tg_pose = np.array([num_joint])

    # retrieve from TG
    input_param = input_action[num_joint:]
    for leg_num in range(len(self._tg)):
        self._tg[leg_num].unpack_params(self.unpack_params(params=input_param, key=leg_num))
        tg_pose[leg_num*num_joint_in_leg : (leg_num+1)*num_joint_in_leg] =  \
            self._tg[leg_num].get_trajectory(self._phi_t)



    return self._pose + input_action[:num_joint] + tg_pose

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation

  def unpack_params(self, params , key=-1):

      num_shared = 1
      num_indie = 9

      if key == -1:
          return params[:num_shared]
      elif key == 0:
          return params[num_shared:(num_shared+num_indie)]
      elif key == 1:
          return params[(num_shared+num_indie):(num_shared+num_indie*2)]
      elif key == 2:
          return params[(num_shared+num_indie*2):(num_shared+num_indie*3)]
      elif key == 3:
          return params[(num_shared+num_indie*3):]
