from motion_imitation.envs.env_wrappers import simple_TG_group
from robots import laikago_pose_utils
from robots import laikago

import numpy as np

trajectory_generator = simple_TG_group.SimpleTGGroup(action_limit=0.2, init_lg_param=None)

current_time = 0
time_increment = 0.01

# copied default params
res = np.zeros([48])


# print(res)

while current_time < 1:
    trajectory_generator.get_action(current_time=current_time, input_action=res)
    # print()
    current_time += time_increment
# trajectory_generator.get_action(current_time=current_time,input_action=res)
