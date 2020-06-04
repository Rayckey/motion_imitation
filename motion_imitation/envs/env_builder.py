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

from envs import locomotion_gym_env
from envs import locomotion_gym_config
from envs.env_wrappers import imitation_wrapper_env
from envs.env_wrappers import observation_dictionary_to_array_wrapper
from envs.env_wrappers import trajectory_generator_wrapper_env
from envs.env_wrappers import simple_openloop
from envs.env_wrappers import simple_TG_group
from envs.env_wrappers import imitation_task
from envs.sensors import environment_sensors
from envs.sensors import sensor_wrappers
from envs.sensors import robot_sensors
from envs.utilities import controllable_env_randomizer_from_config
from robots import laikago
import numpy as np

def build_imitation_env(motion_files, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering):
  assert len(motion_files) > 0

  curriculum_episode_length_start = 20
  curriculum_episode_length_end = 600

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = laikago.Laikago

  sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.BasePositionSensor(), num_history=1),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(['Y', 'R', 'dR', 'P', 'dP']), num_history=1),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=1)
      #, sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
  ]

  task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)

  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            env_randomizers=randomizers, robot_sensors=sensors, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  # env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
  #                                                                      trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND))

  # env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
  #                                                                      trajectory_generator=simple_TG_group.SimpleTGGroup(action_limit=laikago.UPPER_BOUND , init_lg_param=None))
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                         trajectory_generator=simple_TG_group.SimpleTGGroup(
                                                                             action_limit=0.4,
                                                                             init_lg_param=None, is_touting=2, init_f_tg=2))

  # if mode == "test":
  #     curriculum_episode_length_start = curriculum_episode_length_end
  #
  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=30000000,
                                                  num_parallel_envs=num_parallel_envs)
  return env


def build_other_env(motion_files, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering):
    assert len(motion_files) > 0

    curriculum_episode_length_start = 20
    curriculum_episode_length_end = 600

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering

    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    robot_class = laikago.Laikago

    # sensors = [
    #     sensor_wrappers.HistoricSensorWrapper(
    #         wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
    #     sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
    #     sensor_wrappers.HistoricSensorWrapper(
    #         wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
    # ]

    sensors = [
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.BasePositionSensor(), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
    ]



    # Look at this, this is the TG now
    trajectory_generator = simple_TG_group.SimpleTGGroup(
        action_limit=0.4,
        init_lg_param=None, is_touting=2, init_f_tg=2)

    init_lg_param = trajectory_generator.init_lg_param
    # print(" initial tg parameters is this")
    # print(init_lg_param)
    init_lg_param =  init_lg_param[1:]

    tg_init_position = trajectory_generator.get_action(current_time=0, input_action=init_lg_param)


    task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                        enable_cycle_sync=True,
                                        tar_frame_steps=[1, 2, 10, 30],
                                        ref_state_init_prob=0.9,
                                        warmup_time=0.25,tg_init_position=tg_init_position)

    randomizers = []
    if enable_randomizer:
        randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
        randomizers.append(randomizer)

    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                              env_randomizers=randomizers, robot_sensors=sensors, task=task)

    env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)

    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                         trajectory_generator=trajectory_generator)

    if mode == "test":
        curriculum_episode_length_start = curriculum_episode_length_end

    env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                    episode_length_start=curriculum_episode_length_start,
                                                    episode_length_end=curriculum_episode_length_end,
                                                    curriculum_steps=30000000,
                                                    num_parallel_envs=num_parallel_envs)
    return env
