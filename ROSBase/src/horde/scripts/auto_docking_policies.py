#!/usr/bin/env python

"""
Author: Shibhansh Dohare, Niko Yasui.

Description:

Contains the policies needed for auto-docking.

"""
from __future__ import division
import multiprocessing as mp
import numpy as np
import random
from geometry_msgs.msg import Twist, Vector3
import rospy

from policy import Policy
from state_representation import StateConstants
import tools

class eGreedy(Policy):
    """An implementation of eGreedy policy.

    epsilon-Greedy policy, generally used as behaviour policy.   

    """
    def __init__(self, epsilon = 0, *args, **kwargs):
        self.epsilon = epsilon
        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, *args ,**kwargs):
        phi = phi[self.feature_indices]

        q_fun = np.vectorize(lambda action: self.value(phi, action))
        q_values = q_fun(self.action_space)

        best_q = np.max(q_values)
        max_indices = (q_values == best_q)

        self.pi[max_indices] = (1 - self.epsilon) / max_indices.sum()
        self.pi[~max_indices] = 0
        self.pi += self.epsilon / self.action_space.size

class Greedy(Policy):
    """An implementation of Greedy policy.

    Greedy policy, can be used as behaviour and target policy both.   

    """
    def __init__(self, *args, **kwargs):
        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, *args, **kwargs):
        phi = phi[self.feature_indices]

        q_fun = np.vectorize(lambda action: self.value(phi, action))
        q_values = q_fun(self.action_space)

        best_q = np.max(q_values)
        max_indices = (q_values == best_q)

        self.pi[max_indices] = 1 / max_indices.sum()
        self.pi[~max_indices] = 0

class Alternating_Rotation(Policy):
    """An implementation of Alternating Rotation for task-3 of auto-docking.

    According to the policy, the robot rotates in one direction for some time and in the
    opposite direction for some other (align the robot with the docking station).
    It's a non-markov policy.
    Should be used as behavriour policy. 
    It is designed to improve exploration comapred to Greedy or eGreedy for task-3 
    (align the robot with the docking station)

    """
    def __init__(self, *args, **kwargs):
        self.time_steps = 0
        self.num_time_steps = 100
        Policy.__init__(self, *args, **kwargs)

        self.LEFT = 0
        self.RIGHT = 1

    def update(self, *args ,**kwargs):
        if self.time_steps > self.num_time_steps:
            self.last_index = self.LEFT
        else: 
            self.last_index = self.RIGHT

        self.pi = np.zeros(self.action_space.size, float)
        self.pi[self.last_index] = 1

        self.time_steps += 1
        self.time_steps %= 2*self.num_time_steps

class ForwardIfClear(Policy):
    """An implementation of ForwardIfClear for task-2 of auto-docking.

    According to the policy, the robot moves forward with a high probablity and turn with a 
    small probablity. It also turns if the robot was moving forward and it encountered a bump.
    It's a markov process. 
    Should be used as behaviour policy. 
    It is designed to improve exploration comapred to Greedy or eGreedy for task-2 
    (taking the robot in the ceter region created by IR sensors)

    """
    def __init__(self, *args, **kwargs):
        # The last action is recorded according to its respective constants
        # these indices should be in order of action space
        self.FORWARD = 0
        self.TURN_RIGHT = 1
        self.TURN_LEFT = 1
        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        if sum(observation['bump']):
            random_number = random.uniform(0, 1)
            print 'bumping....'
            if random_number > 0.5:
                self.last_index = self.TURN_RIGHT
            else:
                self.last_index = self.TURN_LEFT
        else:
            previous_index = self.last_index
            random_number = random.uniform(0, 1)
            if random_number > 0.95:
                self.last_index = self.TURN_LEFT
            elif random_number > 0.9:
                self.last_index = self.TURN_RIGHT
            else:
                self.last_index = self.FORWARD
        # return 'behavior_policy'
        self.pi = np.zeros(self.action_space.size)
        self.pi[self.last_index] = 1

class Switch:
    """An implementation of Switch for various tasks of auto-docking.

    According to the policy, robot follows an exploring policy for some time steps and then follows the learned
    greedy policy for some time.
    It's a non-markov policy.
    Should be used as behaviour policy. 
    It is designed to check how well the robot has performed at regular intervals.

    """
    def __init__(self, explorer, exploiter, num_timesteps_explore):
        self.explorer = explorer
        self.exploiter = exploiter
        self.num_timesteps_explore = num_timesteps_explore
        self.t = 0

    def update(self, *args, **kwargs):
        self.t += 1
        self.t %= 1.3*self.num_timesteps_explore
        if self.t > self.num_timesteps_explore:
            self.exploiter.update(*args, **kwargs)
            rospy.loginfo('Greedy policy is the behaviour policy, no learning now')
            to_return = 'target_policy'
        else:
            self.explorer.update(*args, **kwargs)
            rospy.loginfo('Explorer policy is the behaviour policy')
            to_return = 'behavior_policy'
        return to_return

    def get_probability(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            prob = self.exploiter.get_probability(*args, **kwargs)
        else:
            prob = self.explorer.get_probability(*args, **kwargs)
        return prob

    def choose_action(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            action = self.exploiter.choose_action(*args, **kwargs)
        else:
            action = self.explorer.choose_action(*args, **kwargs)
        return action