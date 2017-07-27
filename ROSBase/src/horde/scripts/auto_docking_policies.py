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


