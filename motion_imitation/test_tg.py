from motion_imitation.envs.env_wrappers import simple_TG_group
from robots import laikago_pose_utils
from robots import laikago


import numpy as np

trajectory_generator=simple_TG_group.SimpleTGGroup(action_limit=laikago.UPPER_BOUND , init_lg_param=None)


current_time = 0
time_increment = 0.01


# copied default params
f_tg = np.array([2])
gap = np.pi / 2.0
indie = np.array([np.pi / 4, 0.07, 0, 0, 0, -0.30, 0.5, 0, 0.3])
res = np.concatenate([f_tg, indie])
for leg_num in range(1, 4):
    indie[7] += gap
    res = np.concatenate([res, indie])

# insert 0 joint angles
ja = np.zeros([12])
res = np.concatenate([ja, res])

# print(res)

# while current_time < np.pi*3:
#     trajectory_generator.get_action(current_time=current_time,input_action=res)
trajectory_generator.get_action(current_time=current_time,input_action=res)