#Functions for learning policy by ARS

# Importing the libraries
import datetime
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
import os


# Hyper Parameters
class Hp():
    def __init__(self):
        self.nb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'
        self.latent_dim = 2


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
        self.theta_l = np.zeros((output_size, input_size_l + latent_size))
        self.theta_h = np.zeros((latent_size + 1, input_size_h))
        self.theta_size = self.theta_l.size + self.theta_h.size

    '''
    input: all input states with the first successive states being inputs for 
      the high level controller and remaining states being inputs to the low 
      level controller
    '''

    def evaluate(self, input, delta=None, direction=None):
        self.reorderObs(inputs)
        # reshape delta from flat to shape of theta h and l
        if delta is not None:
            [delta_h, delta_l] = self.reshapeFromFlat(delta)
        if direction is None:
            theta_l_temp = self.theta_l
            theta_h_temp = self.theta_h
        elif direction == "positive":
            theta_l_temp = self.theta_l + hp.noise * delta_l
            theta_h_temp = self.theta_h + hp.noise * delta_h
        else:
            theta_l_temp = self.theta_l - hp.noise * delta_l
            theta_h_temp = self.theta_h - hp.noise * delta_h

        input_h = input[0:self.input_size_h]
        input_l = input[self.input_size_h:]
        if self.time_step_h <= 0:
            output_h = theta_h_temp @ input_h
            self.latent_comm = np.clip(output_h[0:self.latent_size], -1, 1)
            self.time_step_h = np.interp(output_h[-1], (-1, 1), (100, 700))
        self.time_step_h -= 1

        return theta_l_temp @ np.concatenate((input_l, self.latent_comm))

    def sampleDeltas(self):
        # samples flat delta
        return [np.random.randn(self.theta_size) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta_size)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d

        # flat theta update values
        update_thetas = hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        [update_h, update_l] = self.reshapeFromFlat(update_thetas)
        self.theta_h += update_h
        self.theta_l += update_l

    def reshapeFromFlat(self, flat):
        h = flat[0:self.theta_h.size].reshape(self.theta_h.shape)
        l = flat[self.theta_h.size:].reshape(self.theta_l.shape)
        return [h, l]


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


# create directory for saving videos of renders
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path