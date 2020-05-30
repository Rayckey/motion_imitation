# Importing the libraries
import datetime
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
from motion_imitation.ars import mkdir
from motion_imitation.ars import Hp
from motion_imitation.ars import HPolicy
from motion_imitation.ars import Normalizer
from motion_imitation.ars import train





import motion_imitation.envs.env_builder as env_builder

# Define constants (maybe read from robot class/environment later)
pmtg_parameter_dim = 1+9*4
leg_pos_dim = 12
action_dim_l = leg_pos_dim+pmtg_parameter_dim
input_dim_h = 4
input_dim_l = 4+leg_pos_dim+1

work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
#update this with our laikago env
env = env_builder.build_imitation_env(motion_files="data/motions/laikago_dog_pace.txt",
                                        num_parallel_envs=1,
                                        mode='train',
                                        enable_randomizer=False,
                                        enable_rendering=True)
env = wrappers.Monitor(env, monitor_dir, force=True)
#nb_inputs = env.observation_space.shape[0]
#nb_outputs = env.action_space.shape[0]
policy = HPolicy(input_dim_h,input_dim_l, hp.latent_dim, action_dim_l)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)