from __future__ import division
import numpy as np
import random
import pickle
import rospy

from state_representation import StateConstants
import tools

class GreedyGQ:
    """ From Maei/Sutton 2010, with additional info from Adam White. """

    def __init__(self, num_features_state_action, action_space, finished_episode, alpha, beta, lmbda, **kwargs):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """

        # parameters
        self.lmbda = lmbda
        self.learning_rate = alpha
        self.secondary_learning_rate = beta
        self.num_features_state_action = num_features_state_action
        self.action_space = action_space
        self.finished_episode = finished_episode

        # learning 
        self.theta = np.zeros(num_features_state_action)
        self.sec_weights = np.zeros(num_features_state_action)
        self.etrace = np.zeros(num_features_state_action)

        # measuring performance
        self.timeStep = 0
        self.average_rewards = [0]
        self.delta = 0
        self.tderr_elig = np.zeros(num_features_state_action)

        # helper
        self.get_state_action = tools.action_state_rep(action_space)
        self.episode_finished_last_step = False

    def predict(self, phi, action):
        if action is not None:
            q = np.dot(self.get_state_action(phi, action), self.theta)
        else:
            get_a_phi = lambda a: self.get_state_action(phi, a)
            dot_fun = np.vectorize(lambda a: np.dot(get_a_phi(a), self.theta))
            q = np.mean(dot_fun(self.action_space))

        return q

    def take_random_action(self):
        random_action = self.action_space[random.randint(0,len(self.action_space)-2)]
        return random_action, 1/len(self.action_space)

    def update(self, phi, last_action, phi_prime, cumulant, gamma, rho, **kwargs):
        self.action_phi = self.get_state_action(phi, last_action)
        self.tderr_elig = self.delta * self.etrace

        # to make sure we don't update anything between the last termination step and the new start step
        # i.e. skip one learning step
        if self.episode_finished_last_step:
            self.episode_finished_last_step = False
            return self.action_phi

        action_phi_primes = {temp_action: self.get_state_action(phi_prime, temp_action) for temp_action in self.action_space}

        action_phis = {temp_action: self.get_state_action(phi, temp_action) for temp_action in self.action_space}

        # self.timeStep = self.timeStep + 1
        # average_reward = self.average_rewards[-1]
        # average_reward = average_reward + (reward - average_reward)/self.timeStepreward

        # self.average_rewards.append(average_reward)

        # if self.timeStep%100 == 0:
        #     with open('average_rewards','w') as f:
        #         pickle.dump(self.average_rewards,f)

        # A_{t+1} update
        next_greedy_action = last_action
        for temp_action in self.action_space:
            if np.dot(self.theta, action_phi_primes[temp_action]) >= np.dot(self.theta, action_phi_primes[next_greedy_action]):
                next_greedy_action = temp_action

        # action_phi_bar update
        action_phi_bar = action_phi_primes[next_greedy_action]

        # delta_t update
        self.td_error = cumulant + gamma * np.dot(self.theta, action_phi_bar) - np.dot(self.theta,self.action_phi)
        
        previous_greedy_action = last_action
        for temp_action in self.action_space:
            if np.dot(self.theta, action_phis[temp_action]) >= np.dot(self.theta, action_phis[previous_greedy_action]):
                previous_greedy_action = temp_action

        if np.count_nonzero(self.theta) == 0:
            rospy.logwarn('self.theta is zero')

        if np.count_nonzero(self.action_phi) == 0:
            rospy.logwarn('self.action_phi is zero')

        # e_t update
        self.etrace *= gamma * self.lmbda * rho
        self.etrace += self.action_phi #(phi_t) 

        if np.count_nonzero(self.etrace) == 0:
            rospy.logwarn('self.eTrace is zero')

        # theta_t update
        self.theta += self.learning_rate * (self.td_error * self.etrace - 
                        gamma * (1 - self.lmbda) * np.dot(self.sec_weights, self.action_phi) * action_phi_bar)

        # temp = self.theta
        # temp = temp/2
        # print np.argsort(temp)

        # if np.count_nonzero(self.theta) == 0:
        #       # theta will be zero in some places before all states are visited
        #     rospy.logwarn('self.theta is zero')
        
        # w_t update
        self.sec_weights += self.secondary_learning_rate * \
            (self.td_error * self.etrace - np.dot(self.sec_weights, self.action_phi) * self.action_phi)

        # for calculating RUPEE
        self.delta = self.td_error

        if self.finished_episode(cumulant):
            rospy.loginfo('Episode finished')
            self.episode_finished_last_step  = True
            self.etrace = np.zeros(self.num_features_state_action)

        # returing to make sure action_phi is used in RUPEE calculation
        return self.action_phi

