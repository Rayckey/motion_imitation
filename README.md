# Hierarchical Locomotion and Path Planning
This repository contains code for training and evaluating hierarchical policies
fo
The code borrows the environment and robot constructions and is adapted
from the code provided alongside the paper:

"Learning Agile Robotic Locomotion Skills by Imitating Animals",

by Xue Bin Peng et al. It provides a Gym environment for training a simulated quadruped robot to imitate various reference motions, and example training code for learning the policies.

Project page: https://xbpeng.github.io/projects/Robotic_Imitation/index.html

## Getting Started

Install dependencies:

- Install requirements: `pip3 install -r requirements.txt`


## Training Models

To train a policy using a single thread, run the following command:

``python3 motion_imitation/rltest.py --mode train --motion_file motion_imitation/data/motions/dog_pace.txt --visualize``

- `--mode` can be either `train`, or `test`, `path`,`sweep` for evaluation
and generation of log files for plotting.
- `--motion_file` specifies the reference motion that the robot uses for pose
initialization. `motion_imitation/data/motions/` contains different reference motion clips.
Here we are not using the full motion, but only take frames for initialization if set.
- `--visualize` enables visualization, and rendering can be disabled by removing the flag.


For parallel training with Ray, run:

`ray start --head --redis-port=6379 --num-cpus=X` where X is the number of CPU's available

- `--n_directions` number of directions to test to obtain reward gradient
- `--deltas_used` number of best directions to update after each iteration
- `--n_workers` number of parallel workers for ARS learning
- `--rollout_length` number of steps per rollout (episode)
- `--currsteps` 0 to turn off curriculum lengthening or curriculum length
- `--actionlim` limit of motor angle corrections [0,1]
- `--saveniters` save weights and log data every n iterations
