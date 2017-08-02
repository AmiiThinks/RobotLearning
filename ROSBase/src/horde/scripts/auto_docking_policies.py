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
    def __init__(self, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        # these indices should be in order of action space
        self.STOP = 0
        self.FORWARD = 1
        self.TURN_RIGHT = 2
        self.TURN_LEFT = 2
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
            if (previous_index == self.TURN_RIGHT or previous_index == self.TURN_LEFT) and self.last_index == self.FORWARD:
                self.last_index = self.STOP
        # return 'behavior_policy'
        self.pi = np.zeros(self.action_space.size)
        self.pi[self.last_index] = 1