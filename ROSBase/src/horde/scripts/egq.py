from __future__ import division
import numpy as np
import random
import sys
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import Twist, Vector3
from matplotlib.pyplot import plot, ion, show
from auto_docking_policies import eGreedy, Learned_Policy
from state_representation import StateConstants
from tools import equal_twists
import pickle

class GreedyGQ:
    """ From Maei/Sutton 2010, with additional info from Adam White. """

    def __init__(self,theta,gamma,_lambda,cumulant,alpha,beta,epsilon,
                learned_policy,num_features_state_action,features_to_use,action_space):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        self.epsilon = epsilon
        self.learned_policy = learned_policy
        self.theta = theta
        self.gamma = gamma
        self._lambda = _lambda
        self.learning_rate = alpha
        self.secondary_learning_rate = beta
        self.cumulant = cumulant
        self.td_error = 0
        self.num_features_state_action = num_features_state_action
        self.sec_weights = np.zeros(num_features_state_action)
        self.etrace = np.zeros(num_features_state_action)
        # remove multiple phi's, we're using phi, rest of the code needs, self.action_phi
        self.action_phi = np.zeros(num_features_state_action)
        self.timeStep = 0
        self.average_rewards = [0]
        self.delta = 0
        self.tderr_elig = np.zeros(num_features_state_action)
        # self.feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        self.action_space = action_space
        self.finished_episode = False

        self.behavior_policy = eGreedy(epsilon = self.epsilon,
                                        theta=self.theta, 
                                        learned_policy=self.learned_policy,
                                        action_space=self.action_space)

    # def take_action(self, phi_prime):
    #     action, mu = self.behavior_policy.take_action(phi = phi_prime,
    #                                                     learned_policy = self.learned_policy,
    #                                                     action_space = self.action_space,
    #                                                     theta = self.theta)
    #     return action, mu

    def  predict(self,x):
        return 0
    def take_random_action(self):
        random_action = self.action_space[random.randint(0,len(self.action_space)-2)]
        return random_action, 1/len(self.action_space)

    def update(self, phi, last_action, observation, phi_prime, **kwargs):
        reward = self.cumulant(observation)
        gamma = self.gamma
        action = last_action
        self.action_phi = self.get_representation(phi,action)

        # to make sure we don't update anything between the last termination step and the new start step
        # i.e. skip one learning step
        if self.finished_episode:
            self.finished_episode = False
            return self.action_phi

        behavior_policy = self.behavior_policy

        print phi
        action_phi_primes = {temp_action: self.get_representation(phi_prime, temp_action) for temp_action in self.action_space}

        action_phis = {temp_action: self.get_representation(phi, temp_action) for temp_action in self.action_space}

        self.timeStep = self.timeStep + 1
        average_reward = self.average_rewards[-1]
        average_reward = average_reward + (reward - average_reward)/self.timeStep

        self.average_rewards.append(average_reward)

        if self.timeStep%100 == 0:
            with open('average_rewards','w') as f:
                pickle.dump(self.average_rewards,f)

        print 'average_reward: ', average_reward

        # A_{t+1} update
        next_greedy_action = action
        for temp_action in self.action_space:
            if np.dot(self.theta, action_phi_primes[temp_action]) >= np.dot(self.theta, action_phi_primes[next_greedy_action]):
                next_greedy_action = temp_action

        # action_phi_bar update
        action_phi_bar = action_phi_primes[next_greedy_action]

        # delta_t update
        self.td_error = reward + gamma * np.dot(self.theta, action_phi_bar) - np.dot(self.theta,self.action_phi)
        
        previous_greedy_action = action
        for temp_action in self.action_space:
            if np.dot(self.theta, action_phis[temp_action]) >= np.dot(self.theta, action_phis[previous_greedy_action]):
                previous_greedy_action = temp_action

        # rho_t (responsibility) update
        if equal_twists(action ,previous_greedy_action):
            responsibility = 1/(1-self.epsilon+self.epsilon/len(self.action_space))
        else:
            responsibility = 0

        print responsibility

        if np.count_nonzero(self.action_phi) == 0:
            print 'self.action_phi is zero'

        # e_t update
        self.etrace *= gamma * self._lambda * responsibility
        self.etrace += self.action_phi #(phi_t) 

        if np.count_nonzero(self.etrace) == 0:
            print 'self.eTrace is zero'
                
        # theta_t update
        self.theta += self.learning_rate * (self.td_error * self.etrace - 
                        gamma * (1 - self._lambda) * np.dot(self.sec_weights, self.action_phi) * action_phi_bar)

        # temp = self.theta
        # temp = temp/2
        # print np.argsort(temp)

        if np.count_nonzero(self.theta) == 0:
            print 'self.theta is zero'
        
        # w_t update
        self.sec_weights += self.secondary_learning_rate * \
            (self.td_error * self.etrace - np.dot(self.sec_weights, self.action_phi) * self.action_phi)

        # for calculating RUPEE
        self.delta = self.td_error
        self.tderr_elig = self.td_error* self.etrace

        if reward == 1:
            print 'Episode finished'
            self.finished_episode = True
            self.etrace = np.zeros(self.num_features_state_action)

        # returing to make sure action_phi is used in RUPEE calculation
        return self.action_phi

    def get_representation(self, state, action):
        representation = []
        state = np.ndarray.tolist(state)
        for index, current_action in enumerate(self.action_space):
            if equal_twists(current_action,action):
                representation = representation + state
            else:
                representation = representation + [0]*len(state)
        return np.asarray(representation)
