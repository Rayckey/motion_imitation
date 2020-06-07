#Functions for learning policy by ARS

# Importing the libraries
import numpy as np
import os
import argparse
import gym.wrappers as wrappers
import ars_multi.logz as logz



#Hyper Parameters
class Hp():
  def __init__(self,
                nb_steps = 1000,
                episode_length = 2000,
                learning_rate = 0.02,
                nb_directions = 16,
                nb_best_directions = 16,
                noise = 0.03,
                seed = 1,
                latent_dim = 2,
                save_weights = True):
    self.nb_steps = nb_steps
    self.episode_length = episode_length
    self.learning_rate = learning_rate
    self.nb_directions = nb_directions
    self.nb_best_directions = nb_best_directions
    assert self.nb_best_directions <= self.nb_directions
    self.noise = noise
    self.seed = seed
    self.latent_dim = latent_dim
    self.save_weights = save_weights


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
  def __init__(self, input_size_h, input_size_l, latent_size, output_size, history_size):
    self.time_step_h = 0;
    self.latent_comm = None;
    self.input_size_h = input_size_h
    self.input_size_l = input_size_l
    self.latent_size = latent_size
    self.theta_l = np.random.uniform(-1,1,size=(output_size, input_size_l + latent_size))
    self.theta_h = np.random.uniform(-1,1,size=(latent_size + 1, input_size_h))
    self.theta_l_bias = np.random.uniform(-1,1,size=output_size)
    self.theta_h_bias = np.random.uniform(-1,1,size=latent_size + 1)
    self.theta_size = self.theta_l.size + self.theta_h.size + self.theta_l_bias.size + self.theta_h_bias.size
    self.history_size = history_size
  '''
  input: all input states with the first successive states being inputs for
    the high level controller and remaining states being inputs to the low
    level controller
  '''
  def evaluate(self, input, delta=None, direction=None):
    #reshape delta from flat to shape of theta h and l
    if delta is not None:
      [delta_h,delta_l,delta_l_bias,delta_h_bias] = self.reshapeFromFlat(delta)
    if direction is None:
      theta_l_temp = self.theta_l
      theta_h_temp = self.theta_h
      theta_l_bias_temp = self.theta_l_bias
      theta_h_bias_temp = self.theta_h_bias
    elif direction == "positive":
      theta_l_temp = self.theta_l + hp.noise * delta_l
      theta_h_temp = self.theta_h + hp.noise * delta_h
      theta_l_bias_temp = self.theta_l_bias + hp.noise * delta_l_bias
      theta_h_bias_temp = self.theta_h_bias + hp.noise * delta_h_bias
    else:
      theta_l_temp = self.theta_l - hp.noise * delta_l
      theta_h_temp = self.theta_h - hp.noise * delta_h
      theta_l_bias_temp = self.theta_l_bias - hp.noise * delta_l_bias
      theta_h_bias_temp = self.theta_h_bias - hp.noise * delta_h_bias

    [input_h,input_l] = self.splitInput(input)
    if self.time_step_h <= 0:
      output_h = np.clip(theta_h_temp@input_h+theta_h_bias_temp,-1,1)
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
    [update_h,update_l,update_l_bias,update_h_bias] = self.reshapeFromFlat(update_thetas)
    self.theta_h += update_h
    self.theta_l += update_l
    self.theta_l_bias += update_l_bias
    self.theta_h_bias += update_h_bias

  def reshapeFromFlat(self, flat):
    h = flat[0:self.theta_h.size].reshape(self.theta_h.shape)
    l = flat[self.theta_h.size:self.theta_h.size+self.theta_l.size].reshape(self.theta_l.shape)
    lb = flat[self.theta_h.size+self.theta_l.size:self.theta_h.size+self.theta_l.size+self.theta_l_bias.size].reshape(self.theta_l_bias.shape)
    hb = flat[self.theta_h.size+self.theta_l.size+self.theta_l_bias.size:].reshape(self.theta_h_bias.shape)
    return [h,l,lb,hb]

  def flattenWeights(self):
      h = np.ndarray.flatten(self.theta_h)
      l = np.ndarray.flatten(self.theta_l)
      lb = np.ndarray.flatten(self.theta_l_bias)
      hb = np.ndarray.flatten(self.theta_h_bias)
      return np.concatenate((h,l,lb,hb))

  def splitInput(self,input):
      h = input[0:3*self.history_size]
      l = input[3*self.history_size+self.history_size*5:]
      for i in range(self.history_size):
        imu_index = int(3*self.history_size+i*5)
        h = np.concatenate([h,input[imu_index,np.newaxis]])
        #the order of IMU history is reversed here (probably okay to keep)
        l = np.concatenate([input[imu_index+1:imu_index+5],l])
      return [h,l]

  def saveWeights(self,step=0):
    weights_path = 'weights_ars'
    if not os.path.exists(weights_path):
            os.makedirs(weights_path)
    np.savetxt('weights_ars/weights_' + str(step) + '.csv', self.flattenWeights(), delimiter=',')

  def loadWeights(self,file_location='weights_ars/weights_0.csv'):
    # save to csv file
    weights = np.loadtxt(file_location, delimiter=',')
    [self.theta_h,self.theta_l,self.theta_l_bias, self.theta_h_bias] = self.reshapeFromFlat(weights)

  def reset(self):
      self.time_step_h = 0
# Two level linear policy class, for use with ARS exploration
class HPolicyhlb():
  '''
  input_size_h: number of inputs for high level controller
  input_size_l: number of sensor and environment inputs for low level
    controller, does not include latent command dimensions
    input_size_h + input_size_l = number of states from environment
  latent_size: hyperparameter, number of dimensions of latent commands
    outputted by high level controller and given to low level controller
  '''
  def __init__(self, input_size_h, input_size_l, latent_size,
                output_size, history_size,latentval1 = None,latentval2 = None):
    self.time_step_h = 0;
    self.latent_comm = None;
    self.input_size_h = input_size_h
    self.input_size_l = input_size_l
    self.latent_size = latent_size
    self.theta_l = np.random.uniform(-1,1,size=(output_size, input_size_l + latent_size))
    self.theta_h = np.random.uniform(-1,1,size=(latent_size + 1, input_size_h))
    self.theta_l_bias = np.random.uniform(-1,1,size=output_size)
    self.theta_size = self.theta_l.size + self.theta_h.size + self.theta_l_bias.size
    self.history_size = history_size
    self.W_mean = 0
    self.W_std = 1
    self.latentval1 = latentval1
    self.latentval2 = latentval2
  '''
  input: all input states with the first successive states being inputs for
    the high level controller and remaining states being inputs to the low
    level controller
  '''
  def evaluate(self, input, delta=None, direction=None, latent1=None, latent2=None):
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

    input = (input - self.W_mean)/self.W_std
    [input_h,input_l] = self.splitInput(input)
    if self.time_step_h <= 0:
      print("NEW COMMAND")
      output_h = np.clip(theta_h_temp@input_h,-1,1)
      if latent1 == None or latent2 == None:
          self.latent_comm = output_h[0:self.latent_size]
      else:
          self.latent_comm = np.asarray([latent1,latent2])
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
    [update_h,update_l,update_l_bias] = self.reshapeFromFlat(update_thetas)
    self.theta_h += update_h
    self.theta_l += update_l
    self.theta_l_bias += update_l_bias

  def reshapeFromFlat(self, flat):
    h = flat[0:self.theta_h.size].reshape(self.theta_h.shape)
    l = flat[self.theta_h.size:self.theta_h.size+self.theta_l.size].reshape(self.theta_l.shape)
    lb = flat[self.theta_h.size+self.theta_l.size:self.theta_h.size+self.theta_l.size+self.theta_l_bias.size].reshape(self.theta_l_bias.shape)
    return [h,l,lb]

  def flattenWeights(self):
      h = np.ndarray.flatten(self.theta_h)
      l = np.ndarray.flatten(self.theta_l)
      lb = np.ndarray.flatten(self.theta_l_bias)
      return np.concatenate((h,l,lb,hb))

  def splitInput(self,input):
      h = input[0:3*self.history_size]
      l = input[3*self.history_size+self.history_size*5:]
      for i in range(self.history_size):
        imu_index = int(3*self.history_size+i*5)
        h = np.concatenate([h,input[imu_index,np.newaxis]])
        #the order of IMU history is reversed here (probably okay to keep)
        l = np.concatenate([input[imu_index+1:imu_index+5],l])
      return [h,l]

  def saveWeights(self,step=0):
    weights_path = 'weights_ars'
    if not os.path.exists(weights_path):
            os.makedirs(weights_path)
    np.savetxt('weights_ars/weights_' + str(step) + '.csv', self.flattenWeights(), delimiter=',')

  def loadWeights(self,file_location='weights_ars/HPolicy.npz'):
    print('loading and building expert policy')
    lin_policy = np.load(file_location, allow_pickle = True)

    lin_policy = lin_policy['arr_0']
    M = lin_policy[0]
    # mean and std of state vectors estimated online by ARS.
    self.W_mean = lin_policy[1]
    self.W_std = lin_policy[2]
    [self.theta_h,self.theta_l,self.theta_l_bias] = self.reshapeFromFlat(M)

  def reset(self):
      self.time_step_h = 0

# Exploring a policy in one specific direction and over one episode
def explore(env, policy, normalizer=None, direction=None, delta=None):
  state = env.reset()
  policy.reset()
  done = False
  num_plays = 0.
  sum_rewards = 0
  while not done and num_plays < hp.episode_length:
    if normalizer != None:
        normalizer.observe(state)
        state = normalizer.normalize(state)
    action = policy.evaluate(state, delta, direction)
    state, reward, done, _ = env.step(action)
    sum_rewards += reward
    num_plays += 1
  return sum_rewards


# Training the AI
def train(env, policy, normalizer, hp):
  max_reward = 0
  saved_rewards = []
  weights_path = 'rewards'
  if not os.path.exists(weights_path):
          os.makedirs(weights_path)
  for step in range(hp.nb_steps):

    # Initializing the perturbations deltas and the positive/negative rewards
    deltas = policy.sampleDeltas()
    positive_rewards = [0] * hp.nb_directions
    negative_rewards = [0] * hp.nb_directions

    # Getting the positive rewards in the positive directions
    for k in range(hp.nb_directions):
      positive_rewards[k] = explore(env, policy, normalizer, direction="positive", delta=deltas[k])

    # Getting the negative rewards in the negative/opposite directions
    for k in range(hp.nb_directions):
      negative_rewards[k] = explore(env, policy, normalizer, direction="negative", delta=deltas[k])

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
    reward_evaluation = explore(env, policy, normalizer)
    saved_rewards.append(reward_evaluation)
    print('Step:', step, 'Reward:', reward_evaluation)

    #save weights for maximum reward and ever 50 step
    if hp.save_weights:
        if reward_evaluation > max_reward:
            max_reward = reward_evaluation
            policy.saveWeights(step=-1)
            np.savetxt('rewards/rewards.csv', saved_rewards, delimiter=',')
        if step%20 is 0:
            policy.saveWeights(step=step)
            np.savetxt('rewards/rewards.csv', saved_rewards, delimiter=',')

#test policy with weights loaded from csv
def test(env, policy, normalizer,weights_file = 'weights_ars/weights_0.csv', max_steps = 10):
    for step in range(max_steps):
        policy.loadWeights(file_location=weights_file)
        reward_evaluation = explore(env, policy, normalizer = None)
        print('Reward:', reward_evaluation)


def sweep(env, policy):
    for l1 in np.linspace(-1,1,10):
        for l2 in np.linspace(-1,1,10):
            for i in range(50):
                state = env.reset()
                policy.reset()
                done = False
                num_plays = 0.
                sum_rewards = 0
                while not done and num_plays < hp.episode_length:
                    action = policy.evaluate(state, latent1=l1, latent2=l2)
                    state, reward, done, _ = env.step(action)
                    logz.log_tabular("l1", l1)
                    logz.log_tabular("l2", l2)
                    logz.log_tabular("x", state[0])
                    logz.log_tabular("y", state[1])
                    logz.dump_tabular()
                    num_plays += 1

def path(env, policy,rollouts = 100):
    for i in range(rollouts):
        state = env.reset()
        policy.reset()
        done = False
        num_plays = 0.
        sum_rewards = 0
        while not done and num_plays < hp.episode_length:
            action = policy.evaluate(state)
            state, reward, done, _ = env.step(action)
            [x,y,_] = env.getxyz()
            logz.log_tabular("x", x)
            logz.log_tabular("y", y)
            logz.dump_tabular()
            num_plays += 1

# MAIN FUCTION
import envs.env_builder as env_builder

# Define constants (maybe read from robot class/environment later)
sensor_history_num = 3
leg_pos_dim = 12
input_dim_h = 4*sensor_history_num
input_dim_l = (4+leg_pos_dim)*sensor_history_num

#update this with our laikago env
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
arg_parser.add_argument("--steps", dest="steps", type=int, default=5000)
arg_parser.add_argument("--eplength", dest="eplength", type=int, default=1000)
arg_parser.add_argument("--learnrate", dest="learnrate", type=float, default=0.02)
arg_parser.add_argument("--ndirections", dest="ndirections", type=int, default=16)
arg_parser.add_argument("--nbestdir", dest="nbestdir", type=int, default=16)
arg_parser.add_argument("--noise", dest="noise", type=float, default=0.03)
arg_parser.add_argument("--latent", dest="latent", type=int, default=2)
arg_parser.add_argument("--weights", dest="weights", type=str, default=None)
arg_parser.add_argument("--saveweights", dest="saveweights", type=bool, default=True)
arg_parser.add_argument("--teststeps", dest="teststeps", type=int, default=10)
arg_parser.add_argument("--actionlim", dest="actionlim", type=float, default=0.2)
arg_parser.add_argument("--savefolder", dest="savefolder", type=str, default="save_data_1")
arg_parser.add_argument("--policytype", dest="policytype", type=int, default=0)
arg_parser.add_argument("--latentval1", dest="latentval1", type=float, default=None)
arg_parser.add_argument("--latentval2", dest="latentval2", type=float, default=None)
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
                                        mode=args.mode,
                                        enable_randomizer=False,
                                        enable_rendering=args.visualize,
                                        action_lim=args.actionlim)

#env = wrappers.Monitor(env, video_path, force=True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
if args.policytype == 0:
    policy = HPolicy(input_dim_h,nb_inputs-input_dim_h,
                    hp.latent_dim, nb_outputs,sensor_history_num)
else:
    policy = HPolicyhlb(input_dim_h,nb_inputs-input_dim_h,
                hp.latent_dim, nb_outputs,sensor_history_num, args.latentval1,args.latentval2)
normalizer = Normalizer(nb_inputs)

if args.mode == 'train':
  if args.weights != None:
      policy.loadWeights(args.weights)
  train(env, policy, normalizer, hp)
elif args.mode == 'test':
  test(env, policy, normalizer, args.weights, args.teststeps)
elif args.mode == 'sweep':
    if args.weights != None:
        policy.loadWeights(args.weights)
    dir_path = 'data'
    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    logz.configure_output_dir(logdir)
    sweep(env,policy)
elif args.mode == 'path':
    if args.weights != None:
        policy.loadWeights(args.weights)
    dir_path = 'data'
    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    logz.configure_output_dir(logdir)
    path(env,policy)
