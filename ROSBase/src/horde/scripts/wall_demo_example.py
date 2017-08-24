"""Runs the demo that learns to avoid walls.

This module describes an agent that learns to avoid walls. It specifies
the agent's learning algorithm, parameters, policies, features, and
actions. The module also interfaces with the :doc:`learning_foreground`
and the :doc:`action_manager` to run the main learning loop and publish
actions respectively.

All parameters are set in ``if __name__ == "__main__"``

Authors:
    Michele Albach, Shibhansh Dohare, Banafsheh Rafiee,
    Parash Rahman, Niko Yasui.
"""
from __future__ import division

import multiprocessing as mp
from multiprocessing import Value
import random

import numpy as np
import rospy
from geometry_msgs.msg import Twist, Vector3

from action_manager import start_action_manager
from gtd import GTD
from gvf import GVF
from learning_foreground import start_learning_foreground
from policy import Policy
from state_representation import StateConstants
import tools


class GoForward(Policy):
    """Target Policy.

    Constant policy that only goes forward.

    Attributes:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
        fwd_action_index (int): Index of ``action_space`` containing the
            forward action.
    """

    def __init__(self, action_space, fwd_action_index, *args, **kwargs):
        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)

        self.pi *= 0.0
        self.pi[fwd_action_index] += 1.0

    def update(self, phi, observation, *args, **kwargs):
        pass


class PavlovSoftmax(Policy):
    """Behavior Policy.

    Softmax policy that forces the agent to select a "turn"
    action if the bump sensor is on.

    Attributes:
        time_scale (float): Number of seconds in a learning timestep.
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
        value_function: A function used by the policy to update values
            of pi. This is usually a value function learned by a GVF.
        feature_indices (numpy array of bool): Indices of the feature
            vector corresponding to indices used by the
            :py:obj:`value_function`.
    """

    def __init__(self,
                 time_scale,
                 action_space,
                 value_function,
                 feature_indices,
                 *args, **kwargs):
        self.TURN = 1
        self.FORWARD = 0

        self.time_scale = time_scale

        kwargs['action_space'] = action_space
        kwargs['value_function'] = value_function
        kwargs['feature_indices'] = feature_indices
        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        """Updates :py:attr:`~policy.Policy.pi`."""
        phi = phi[self.feature_indices]
        p = self.value_function(phi)

        if observation['bump']:
            self.pi *= 0
            self.pi[self.TURN] = 1
        else:
            # Joseph Modayil's constants
            # T = 1 / self.time_scale / 1
            T = 6.0
            k1 = np.log((T - 1) * (self.action_space.size - 1))
            k2 = k1 * 4.0

            # make preferences for each action
            prefs = np.zeros(2)

            def last(index):
                return self.last_index == index
            prefs[self.TURN] = k1 * last(self.TURN)
            prefs[self.FORWARD] = k2 * (0.5 - p) + k1 * last(self.FORWARD)

            self.pi = tools.softmax(prefs)


if __name__ == "__main__":
    try:
        random.seed(20170823)

        # turns on and off the hyperparameter search
        hyperparameter_experiment_mode = False

        action_manager_process = mp.Process(target=start_action_manager,
                                            name="action_manager",
                                            args=())
        action_manager_process.start()

        # robotic parameters
        time_scale = 0.1
        forward_speed = 0.12
        turn_speed = 5. / 3

        # all available actions
        action_space = np.array([Twist(Vector3(forward_speed, 0, 0),
                                       Vector3(0, 0, 0)),
                                 Twist(Vector3(0, 0, 0),
                                       Vector3(0, 0, turn_speed))])

        # learning parameters
        alpha0 = 0.05
        lmbda = 0.9
        discount = 0.97

        features_to_use = {'image', 'bias'}
        print_stats = ['cumulant', 'prediction']

        feature_indices = np.concatenate(
                [StateConstants.indices_in_phi[f] for f in
                 features_to_use])
        num_active_features = sum(
                StateConstants.num_active_features[f] for f in
                features_to_use)
        num_features = feature_indices.size

        def discount_if_bump(obs):
            return 0 if obs["bump"] else discount

        def one_if_bump(obs):
            return int(obs['bump']) if obs is not None else 0

        dtb_hp = {'alpha': alpha0 / num_active_features,
                  'beta': alpha0 / 1000 / num_active_features,
                  'lmbda': lmbda,
                  'alpha0': alpha0,
                  'num_features': num_features,
                  'feature_indices': feature_indices,
                  }

        # prediction GVF
        dtb_policy = GoForward(action_space=action_space,
                               fwd_action_index=0)
        dtb_learner = GTD(**dtb_hp)

        threshold_behavior_policy = PavlovSoftmax(
                                action_space=action_space,
                                feature_indices=dtb_hp['feature_indices'],
                                value_function=dtb_learner.predict,
                                time_scale=time_scale)
        distance_to_bump = GVF(cumulant=one_if_bump,
                               gamma=discount_if_bump,
                               target_policy=dtb_policy,
                               learner=dtb_learner,
                               name='DistanceToBump',
                               **dtb_hp)

        # start processes
        cumulant_counter = Value('d', 0)
        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [distance_to_bump],
                                              features_to_use,
                                              threshold_behavior_policy,
                                              print_stats,
                                              None,
                                              cumulant_counter))

        foreground_process.start()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
    finally:
        try:
            foreground_process.join()
            action_manager_process.join()
        except NameError:
            pass
