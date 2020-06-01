#Functions for learning policy by ARS

# Importing the libraries
import numpy as np
import os
import argparse
import gym.wrappers as wrappers
from numpy import savetxt


#Hyper Parameters
class Hp():
  def __init__(self,
                nb_steps = 1000,
                episode_length = 1000,
                learning_rate = 0.02,
                nb_directions = 16,
                nb_best_directions = 16,
                noise = 0.03,
                seed = 1,
                latent_dim = 2):
    self.nb_steps = nb_steps
    self.episode_length = episode_length
    self.learning_rate = learning_rate
    self.nb_directions = nb_directions
    self.nb_best_directions = nb_best_directions
    assert self.nb_best_directions <= self.nb_directions
    self.noise = noise
    self.seed = seed
    self.latent_dim = latent_dim


# Normalizing the states
class Normalizer():
  def __init__(self, nb_inputs):
    self.n = np.zeros(nb_inputs)
    self.mean = np.zeros(nb_inputs)
    self.mean_diff = np.zeros(nb_inputs)
    self.var = np.zeros(nb_inputs)

  def observe(self, x):
    self.n += 1.
    last_mean = self.mean.copy()
    self.mean += (x - self.mean) / self.n
    self.mean_diff += (x - last_mean) * (x - self.mean)
    self.var = (self.mean_diff / self.n).clip(min=1e-2)

  def normalize(self, inputs):
    obs_mean = self.mean
    obs_std = np.sqrt(self.var)
    return (inputs - obs_mean) / obs_std


# Two level linear policy class, for use with ARS exploration
class HPolicy():
  '''
  input_size_h: number of inputs for high level controller
  input_size_l: number of sensor and environment inputs for low level
    controller, does not include latent command dimensions
    input_size_h + input_size_l = number of states from environment
  latent_size: hyperparameter, number of dimensions of latent commands
    outputted by high level controller and given to low level controller
  '''
  def __init__(self, input_size_h, input_size_l, latent_size, output_size):
    self.time_step_h = 0;
    self.latent_comm = None;
    self.input_size_h = input_size_h
    self.input_size_l = input_size_l
    self.latent_size = latent_size
    self.theta_l = np.random.uniform(-1,1,size=(output_size, input_size_l + latent_size))
    self.theta_h = np.random.uniform(-1,1,size=(latent_size + 1, input_size_h))
    self.theta_l_bias = np.random.uniform(-1,1,size=output_size)
    self.theta_size = self.theta_l.size + self.theta_h.size + self.theta_l_bias.size
  '''
  input: all input states with the first successive states being inputs for
    the high level controller and remaining states being inputs to the low
    level controller
  '''
  def evaluate(self, input, delta=None, direction=None):
    #reshape delta from flat to shape of theta h and l
    if delta is not None:
      [delta_h,delta_l,delta_l_bias] = self.reshapeFromFlat(delta)
    if direction is None:
      theta_l_temp = self.theta_l
      theta_h_temp = self.theta_h
      theta_l_bias_temp = self.theta_l_bias
    elif direction == "positive":
      theta_l_temp = self.theta_l + hp.noise * delta_l
      theta_h_temp = self.theta_h + hp.noise * delta_h
      theta_l_bias_temp = self.theta_l_bias + hp.noise * delta_l_bias
    else:
      theta_l_temp = self.theta_l - hp.noise * delta_l
      theta_h_temp = self.theta_h - hp.noise * delta_h
      theta_l_bias_temp = self.theta_l_bias - hp.noise * delta_l_bias

    input_h = input[0:self.input_size_h]
    input_l = input[self.input_size_h:]
    if self.time_step_h <= 0:
      output_h = np.clip(theta_h_temp@input_h,-1,1)
      self.latent_comm = output_h[0:self.latent_size]
      self.time_step_h = np.interp(output_h[-1], (-1, 1), (100, 700))
    self.time_step_h -= 1
    return theta_l_temp@np.concatenate((input_l, self.latent_comm))+theta_l_bias_temp

  def sampleDeltas(self):
    #samples flat delta
    return [np.random.randn(self.theta_size) for _ in range(hp.nb_directions)]

  def update(self, rollouts, sigma_r):
    step = np.zeros(self.theta_size)
    for r_pos, r_neg, d in rollouts:
      step += (r_pos - r_neg) * d

    #flat theta update values
    update_thetas = hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
    [update_h,update_l,update_bias] = self.reshapeFromFlat(update_thetas)
    self.theta_h += update_h
    self.theta_l += update_l
    self.theta_l_bias += update_bias

  def reshapeFromFlat(self, flat):
    h = flat[0:self.theta_h.size].reshape(self.theta_h.shape)
    l = flat[self.theta_h.size:self.theta_h.size+self.theta_l.size].reshape(self.theta_l.shape)
    b = flat[self.theta_h.size+self.theta_l.size:].reshape(self.theta_l_bias.shape)
    return [h,l,b]

  def flattenWeights(self):
      h = np.ndarray.flatten(self.theta_h)
      l = np.ndarray.flatten(self.theta_l)
      b = np.ndarray.flatten(self.theta_l_bias)
      return np.concatenate((h,l,b))

  def saveWeights(self,step=0):
    weights_path = 'weights_ars'
    if not os.path.exists(weights_path):
            os.makedirs(weights_path)
    savetxt('weights_ars/weights' + str(step) + '.csv', self.flattenWeights(), delimiter=',')

  def loadWeights(self,file_location='weights_ars/weights_0.csv'):
    # save to csv file
    weights = loadtxt(file_location, delimiter=',')
    [self.theta_h,self.theta_l,self.theta_l_bias] = reshapeFromFlat(weights)

# Exploring a policy in one specific direction and over one episode
def explore(env, normalizer, policy, direction=None, delta=None):
  state = env.reset()
  done = False
  num_plays = 0.
  sum_rewards = 0
  while not done and num_plays < hp.episode_length:
    normalizer.observe(state)
    state = normalizer.normalize(state)
    action = policy.evaluate(state, delta, direction)
    state, reward, done, _ = env.step(action)

    sum_rewards += reward
    num_plays += 1
  return sum_rewards


# Training the AI
def train(env, policy, normalizer, hp):
  for step in range(hp.nb_steps):

    # Initializing the perturbations deltas and the positive/negative rewards
    deltas = policy.sampleDeltas()
    positive_rewards = [0] * hp.nb_directions
    negative_rewards = [0] * hp.nb_directions

    # Getting the positive rewards in the positive directions
    for k in range(hp.nb_directions):
      positive_rewards[k] = explore(env, normalizer, policy, direction="positive", delta=deltas[k])

    # Getting the negative rewards in the negative/opposite directions
    for k in range(hp.nb_directions):
      negative_rewards[k] = explore(env, normalizer, policy, direction="negative", delta=deltas[k])

    # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
    all_rewards = np.array(positive_rewards + negative_rewards)
    sigma_r = all_rewards.std()

    # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
    scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
    order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:hp.nb_best_directions]
    rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

    # Updating our policy
    policy.update(rollouts, sigma_r)

    # Printing the final reward of the policy after the update
    reward_evaluation = explore(env, normalizer, policy)
    print('Step:', step, 'Reward:', reward_evaluation)

    #save weights
    if step%10 is 0:
      policy.saveWeights(step=step)

#test policy with weights loaded from csv
def test(env, policy, normalizer,weights_file = 'weights_ars/weights_0.csv'):
    policy.loadWeights(file_location=weights_file)
    reward_evaluation = explore(env, normalizer, policy)
    print('Step:', step, 'Reward:', reward_evaluation)


# MAIN FUCTION
import envs.env_builder as env_builder

# Define constants (maybe read from robot class/environment later)
sensor_history_num = 1
leg_pos_dim = 12
input_dim_h = 4*sensor_history_num
input_dim_l = (4+leg_pos_dim)*sensor_history_num

#update this with our laikago env
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
arg_parser.add_argument("--steps", dest="steps", type=int, default=1000)
arg_parser.add_argument("--eplength", dest="eplength", type=int, default=1000)
arg_parser.add_argument("--learnrate", dest="learnrate", type=float, default=0.02)
arg_parser.add_argument("--ndirections", dest="ndirections", type=int, default=16)
arg_parser.add_argument("--nbestdir", dest="nbestdir", type=int, default=16)
arg_parser.add_argument("--noise", dest="noise", type=float, default=0.03)
arg_parser.add_argument("--latent", dest="latent", type=int, default=2)
arg_parser.add_argument("--weights", dest="weights", type=str, default='weights_ars/weights_0.csv')

args = arg_parser.parse_args()

hp = Hp(nb_steps = args.steps,
    episode_length = args.eplength,
    learning_rate = args.learnrate,
    nb_directions = args.ndirections,
    nb_best_directions = args.nbestdir,
    noise = args.noise,
    seed = 1,
    latent_dim = args.latent)
np.random.seed(hp.seed)

env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=1,
                                        mode='train',
                                        enable_randomizer=False,
                                        enable_rendering=args.visualize)

#env = wrappers.Monitor(env, video_path, force=True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = HPolicy(input_dim_h,input_dim_l, hp.latent_dim, nb_outputs)
normalizer = Normalizer(nb_inputs)

if args.mode is "train":
  train(env, policy, normalizer, hp)
else:
  test(env, policy, normalizer, args.weights)
