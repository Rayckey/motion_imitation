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

"""Contains the terminal conditions for imitation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from utilities import pose3d
from utilities import motion_util
from pybullet_utils import transformations


def imitation_terminal_condition(env,
                                 goal = np.array([0,0,0.48]),
                                 dist_to_goal = 0.2,
                                 dist_fail_threshold=1.0,
                                 rot_fail_threshold=0.5 * np.pi,
                                 path=0):
  """A terminal condition for motion imitation task.

  Args:
    env: An instance of MinitaurGymEnv
    dist_fail_threshold: Max distance the simulated character's root is allowed
      to drift from the reference motion before the episode terminates.
    rot_fail_threshold: Max rotational difference between simulated character's
      root and the reference motion's root before the episode terminates.
    dist_to_goal: acceptable euclidean distance to goal to be considered complete
  Returns:
    A boolean indicating if episode is over.
  """

  pyb = env._pybullet_client
  task = env._task

  motion_over = task.is_motion_over()
  foot_links = env.robot.GetFootLinkIDs()
  ground = env.get_ground()

  contact_fall = False
  # sometimes the robot can be initialized with some ground penetration
  # so do not check for contacts until after the first env step.
  if env.env_step_counter > 0:
    robot_ground_contacts = env.pybullet_client.getContactPoints(
        bodyA=env.robot.quadruped, bodyB=ground)

    for contact in robot_ground_contacts:
      if contact[3] not in foot_links:
        contact_fall = True
        break

  # root_pos_ref, root_rot_ref = pyb.getBasePositionAndOrientation(
  #     task.get_ref_model())
  root_pos_sim, root_rot_sim = pyb.getBasePositionAndOrientation(
      env.robot.quadruped)

  # root_pos_diff = np.array(root_pos_ref) - np.array(root_pos_sim)
  # root_pos_fail = (
  #     root_pos_diff.dot(root_pos_diff) >
  #     dist_fail_threshold * dist_fail_threshold)
  #
  # root_rot_diff = transformations.quaternion_multiply(
  #     np.array(root_rot_ref),
  #     transformations.quaternion_conjugate(np.array(root_rot_sim)))
  # _, root_rot_diff_angle = pose3d.QuaternionToAxisAngle(
  #     root_rot_diff)
  # root_rot_diff_angle = motion_util.normalize_rotation_angle(
  #     root_rot_diff_angle)
  # root_rot_fail = (np.abs(root_rot_diff_angle) > rot_fail_threshold)

  #check if at goal by distance to goal
  at_goal = np.linalg.norm(root_pos_sim-goal) < dist_to_goal

  #define path here (ex parabola +/- 1)
  [x_pos,y_pos,z_pos] = root_pos_sim

  if path == 1:
      f = 0.1*x_pos**2-x_pos
  elif path == 2:
      f = -0.1*x_pos**2+x_pos
  else:
      f = -0.07*x_pos**2+0.7*x_pos
  out_of_path = y_pos < f-1 or y_pos > f+1

  if path == -1:
      out_of_path = false
  elif path == 3:
      out_of_path = np.sqrt(y_pos**2+x_pos**2) > 10
  done = contact_fall \
        or at_goal \
        or out_of_path
      # or motion_over \
      # or root_pos_fail \
      # or root_rot_fail \


  return done
