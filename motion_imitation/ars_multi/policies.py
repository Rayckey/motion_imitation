'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''


import numpy as np
from ars_multi.filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):

        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

class HLinearPolicy(Policy):
    """
    Hierarchical policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)

        self.latent_dim = policy_params['latent_dim']
        self.ob_h_dim = policy_params['ob_h_dim']
        self.ob_l_dim = policy_params['ob_l_dim']
        self.history_size = policy_params['history_size']
        self.time_step_h = 0
        self.latent_comm = None
        self.theta_h_size = (self.latent_dim + 1) * self.ob_h_dim
        self.theta_l_size = self.ac_dim*(self.latent_dim+self.ob_l_dim)
        self.theta_size = self.ac_dim * (self.ob_l_dim + self.latent_dim + 1) + self.theta_h_size
        self.weights = np.zeros(self.theta_size, dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        #reshape delta from flat to shape of theta h and l
        [theta_h,theta_l,theta_b] = self.reshapeFromFlat(self.weights)

        [ob_h,ob_l] = self.splitInput(ob)
        #high level policy
        if self.time_step_h <= 0:
          output_h = np.clip(theta_h@ob_h,-1,1)
          self.latent_comm = output_h[0:self.latent_dim]
          self.time_step_h = np.interp(output_h[-1], (-1, 1), (20, 200))
        self.time_step_h -= 1
        return theta_l@np.concatenate((ob_l, self.latent_comm))+theta_b


    def reshapeFromFlat(self, flat):
        h = flat[0:self.theta_h_size].reshape((self.latent_dim + 1,self.ob_h_dim))
        l = flat[self.theta_h_size:self.theta_h_size+self.theta_l_size].reshape((self.ac_dim,(self.latent_dim+self.ob_l_dim)))
        b = flat[self.theta_h_size+self.theta_l_size:].reshape((self.ac_dim,))
        return [h,l,b]

    def get_weights_plus_stats(self):

        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def splitInput(self,input):
        h = input[0:3*self.history_size]
        l = input[3*self.history_size+self.history_size*5:]
        for i in range(self.history_size):
          imu_index = int(3*self.history_size+i*5)
          h = np.concatenate([h,input[imu_index,np.newaxis]])
          #the order of IMU history is reversed here (probably okay to keep)
          l = np.concatenate([input[imu_index+1:imu_index+5],l])
        return [h,l]

    def loadWeights(self,filepath):
        print('loading policy')
        lin_policy = np.load(filepath, allow_pickle = True)
        lin_policy = lin_policy['arr_0']
        self.weights = lin_policy[0]

    def reset(self):
        self.time_step_h = 0

class HLinearPolicyHOnly(Policy):
    """
    Hierarchical policy class that computes action as <w, ob>.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)

        self.latent_dim = policy_params['latent_dim']
        self.ob_h_dim = policy_params['ob_h_dim']
        self.ob_l_dim = policy_params['ob_l_dim']
        self.history_size = policy_params['history_size']
        self.time_step_h = 0
        self.latent_comm = None
        self.theta_h_size = (self.latent_dim + 1) * self.ob_h_dim
        self.theta_l_size = self.ac_dim*(self.latent_dim+self.ob_l_dim)
        self.theta_size = self.ac_dim * (self.ob_l_dim + self.latent_dim + 1) + self.theta_h_size
        self.lweights = np.zeros(self.theta_l_size + self.ac_dim, dtype = np.float64)
        self.weights = np.zeros(self.theta_h_size, dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        #reshape delta from flat to shape of theta h and l
        [theta_h,theta_l,theta_b] = self.reshapeFromFlat(self.weights,self.lweights)

        [ob_h,ob_l] = self.splitInput(ob)
        #high level policy
        if self.time_step_h <= 0:
          output_h = np.clip(theta_h@ob_h,-1,1)
          self.latent_comm = output_h[0:self.latent_dim]
          self.time_step_h = np.interp(output_h[-1], (-1, 1), (20, 200))
        self.time_step_h -= 1
        return theta_l@np.concatenate((ob_l, self.latent_comm))+theta_b


    def reshapeFromFlat(self, h,l):
        h = h.reshape((self.latent_dim + 1,self.ob_h_dim))
        l = l[:self.theta_l_size].reshape((self.ac_dim,(self.latent_dim+self.ob_l_dim)))
        b = l[self.theta_l_size:].reshape((self.ac_dim,))
        return [h,l,b]

    def get_weights_plus_stats(self):

        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def splitInput(self,input):
        h = input[0:3*self.history_size]
        l = input[3*self.history_size+self.history_size*5:]
        for i in range(self.history_size):
          imu_index = int(3*self.history_size+i*5)
          h = np.concatenate([h,input[imu_index,np.newaxis]])
          #the order of IMU history is reversed here (probably okay to keep)
          l = np.concatenate([input[imu_index+1:imu_index+5],l])
        return [h,l]

    def loadWeights(self,filepath):
        print('loading policy')
        lin_policy = np.load(filepath, allow_pickle = True)
        lin_policy = lin_policy['arr_0']
        self.weights = lin_policy[0][:self.theta_h_size]
        self.lweights = lin_policy[0][self.theta_h_size:]

    def reset(self):
        self.time_step_h = 0
