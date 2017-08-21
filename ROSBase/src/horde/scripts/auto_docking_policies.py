#!/usr/bin/env python

"""Contains the policies needed for auto-docking.

Authors:
    Shibhansh Dohare, Niko Yasui.
"""
from __future__ import division

import numpy as np
import rospy

from policy import Policy


class EGreedy(Policy):
    """An epsilon-greedy policy.

    Args:
        epsilon (float, optional): Proportion of time to take a random
            action. Default: 0 (greedy).
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
        value_function: A function used by the Policy to update values 
            of pi. This is usually a value function learned by a GVF.
        feature_indices (numpy array of bool): Indices of the feature
            vector corresponding to indices used by the
            :py:obj:`value_function`.
    """
    def __init__(self, 
                 epsilon,
                 action_space, 
                 value_function,
                 feature_indices,
                 *args, **kwargs):
        self.epsilon = epsilon

        self.value = value_function

        kwargs['action_space'] = action_space
        kwargs['value_function'] = value_function
        kwargs['feature_indices'] = feature_indices
        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, *args, **kwargs):
        phi = phi[self.feature_indices]

        q_fun = np.vectorize(lambda action: self.value_function(phi, action))
        q_values = q_fun(self.action_space)

        best_q = np.max(q_values)
        max_indices = (q_values == best_q)

        self.pi[max_indices] = (1 - self.epsilon) / max_indices.sum()
        self.pi[~max_indices] = 0
        self.pi += self.epsilon / self.action_space.size


class AlternatingRotation(Policy):
    """A policy for task-3 of auto-docking.

    According to the policy, the robot rotates in one direction for some
    time and in the opposite direction for some other time, with the
    goal of aligning the robot with the docking station.
    It's a non-markov policy.
    Should be used as behaviour policy. 
    It is designed to improve exploration comapred to Greedy or eGreedy
    for task-3 (align the robot with the docking station).

    Args:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
    """

    def __init__(self, action_space, *args, **kwargs):
        self.time_steps = 0
        self.num_time_steps = 100

        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)

        self.LEFT = 0
        self.RIGHT = 1

    def update(self, *args, **kwargs):
        """Deterministically sets ``pi`` based on the timestep.
        """
        if self.time_steps > self.num_time_steps:
            self.last_index = self.LEFT
        else:
            self.last_index = self.RIGHT

        self.pi *= 0
        self.pi[self.last_index] = 1

        self.time_steps += 1
        self.time_steps %= 2 * self.num_time_steps


class ForwardIfClear(Policy):
    """An implementation of ForwardIfClear for task-2 of auto-docking.

    According to the policy, the robot moves forward with a high
    probablity and turns with a small probablity. It also turns if the
    robot was moving forward and it encountered a bump.
    It's a markov policy. 
    Should be used as behaviour policy. 
    It is designed to improve exploration comapred to Greedy or eGreedy
    for task-2 (taking the robot to the center IR region).
    """

    def __init__(self, action_space, *args, **kwargs):
        # The last action is recorded according to its respective constants
        # these indices should be in order of action space
        self.FORWARD = 0
        self.TURN_RIGHT = 2
        self.TURN_LEFT = 1

        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        """Updates ``pi`` depending on if there is a bump or not."""
        self.pi *= 0
        if observation['bump']:
            self.pi[self.TURN_LEFT] += 0.5
            self.pi[self.TURN_RIGHT] += 0.5
        else:
            self.pi += 0.05
            self.pi[self.FORWARD] += 0.9


class Switch:
    """Switches between two policies.

    According to the policy, robot follows an exploring policy for some
    time steps and then follows the learned greedy policy for some time.
    It's a non-markov policy.
    Should be used as behaviour policy. 
    It is designed to check how well the robot has performed at
    regular intervals.

    Attributes:
        explorer (policy): Policy to use for exploration.
        exploiter (policy): Policy to use for exploitation.
        num_timesteps_explore (int): Number of timesteps to run each
            policy before switching.
        t (int, not passed): Counter. Switch policies when counter
            reaches ``num_timesteps_explore``. 

    """

    def __init__(self, explorer, exploiter, num_timesteps_explore):
        self.explorer = explorer
        self.exploiter = exploiter
        self.num_timesteps_explore = num_timesteps_explore
        self.t = 0

        self.action_space = self.explorer.action_space
        self.last_index = self.explorer.last_index
    def update(self, *args, **kwargs):
        self.t += 1
        self.t %= self.num_timesteps_explore
        if self.t > self.num_timesteps_explore:
            self.exploiter.update(*args, **kwargs)
            rospy.loginfo(
                'Greedy policy is the behaviour policy, no learning now')
            to_return = 'target_policy'

            self.last_index = self.exploiter.last_index
        else:
            self.explorer.update(*args, **kwargs)
            rospy.loginfo('Explorer policy is the behaviour policy')
            to_return = 'behavior_policy'

            self.last_index = self.explorer.last_index
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
