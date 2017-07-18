from __future__ import division
import numpy as np
import random
import sys
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import Twist, Vector3
from matplotlib.pyplot import plot, ion, show
from auto_docking_policies import eGreedy, Learned_Policy

class GreedyGQ:
    """ From Maei/Sutton 2010, with additional info from Adam White. """

    def __init__(self,theta,gamma,_lambda,cumulant,alpha,beta,epsilon,learned_policy):
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
        self.sec_weights = np.zeros(14404*5)
        self.etrace = np.zeros(14404*5)
        # remove multiple phi's, we're using phi, rest of the code needs, self.phi
        self._phi = np.zeros(14404)
        self.phi = np.zeros(14404*5)
        self.timeStep = 0
        self.average_rewards = [0]
        self.delta = 0
        self.tderr_elig = np.zeros(14404*5)
        self.action_space = [Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                        Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0)), # forward
                        Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0)), # backward
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, 1.5)), # turn acw/cw
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, -1.5)) # turn cw/acw
                        ]

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

    def update(self, state, action, observation, next_state, **kwargs):
        reward = self.cumulant(observation)
        gamma = self.gamma
        learned_policy = self.learned_policy
        behavior_policy = self.behavior_policy
        self.phi = self.get_representation(state,action)
        self.timeStep = self.timeStep + 1
        average_reward = self.average_rewards[-1]
        average_reward = average_reward + (reward - average_reward)/self.timeStep

        self.average_rewards.append(average_reward)

        print 'average_reward: ', average_reward

        # A_{t+1} update
        next_greedy_action = action
        for action in self.action_space:
            if np.dot(self.theta, self.get_representation(next_state,action)) >= np.dot(self.theta, self.get_representation(next_state,next_greedy_action)):
                next_greedy_action = action

        # phi_bar update
        phi_bar = self.get_representation(next_state,next_greedy_action)

        # delta_t update
        self.td_error = reward + gamma * np.dot(self.theta, phi_bar) - np.dot(self.theta,self.phi)
        
        previous_greedy_action = action
        for temp_action in self.action_space:
            if np.dot(self.theta, self.get_representation(state,temp_action)) >= np.dot(self.theta, self.get_representation(state,previous_greedy_action)):
                previous_greedy_action = temp_action

        # rho_t (responsibility) update
        if equal_twists(action ,previous_greedy_action):
            responsibility = 1/(1-self.epsilon+self.epsilon/len(self.action_space))
        else:
            responsibility = 0

        if np.count_nonzero(self.phi) == 0:
            print 'self.phi is zero'

        # e_t update
        self.etrace *= gamma * self._lambda * responsibility
        self.etrace += self.phi #(phi_t) 

        if np.count_nonzero(self.etrace) == 0:
            print 'self.eTrace is zero'
                
        # theta_t update
        self.theta += self.learning_rate * (self.td_error * self.etrace - 
                        gamma * (1 - self._lambda) * np.dot(self.sec_weights, self.phi) * phi_bar)

        if np.count_nonzero(self.theta) == 0:
            print 'self.theta is zero'
        
        # w_t update
        self.sec_weights += self.secondary_learning_rate * (self.td_error * self.etrace - np.dot(self.sec_weights, self.phi) * self.phi)
        self.delta = self.td_error
        self.tderr_elig = self.td_error* self.eTrace

        if reward == 1:
            print 'Episode finished'
            self.etrace = np.zeros(14404*5)


    def get_representation(self, state, action):
        representation = []
        state = np.ndarray.tolist(state)
        for index, current_action in enumerate(self.action_space):
            if equal_twists(current_action,action):
                representation = representation + state
            else:
                representation = representation + [0]*len(state)
        return np.asarray(representation)
