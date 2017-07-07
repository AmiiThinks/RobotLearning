"""
Author: Michele Albach, David Quail, Parash Rahman, Niko Yasui, June 2017.

Description:
Behavior policy is responsible for returning the desired action to take.
"""
import geometry_msgs.msg as geom_msg
import random

from tools import equal_twists

class Policy:
    def __init__(self):
        pass

    def __call__(self, phi, observation):
        """
        Implements the policy.

        Input:
            state - state object as defined in learning foreground

        Output:
            action - Twist object
            prob - probability of taking the action
        """

        # build Twist object
        action = geom_msg.Twist()

        # probability of taking the action
        prob = 1

        return action, prob

    def prob(self, action, phi, observation):
        # replace this with stochastic code for a non-deterministic policy
        return self.__call__(phi, observation)[1]


