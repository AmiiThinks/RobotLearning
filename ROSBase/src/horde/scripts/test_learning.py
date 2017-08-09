"""Predicts spinning speed.

All parameters are set in ``if __name__ == "__main__"``

Authors:
    Niko Yasui.
"""
from __future__ import division
import math
import multiprocessing as mp

from geometry_msgs.msg import Twist, Vector3
import numpy as np
import rospy

from action_manager import start_action_manager
from gtd import GTD
from gvf import GVF
from learning_foreground import start_learning_foreground
from policy import Policy
from state_representation import StateConstants
import tools

class Spin(Policy):
    """Constant policy that only spins.

    Attributes:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
        turn_action_index (int): Index of ``action_space`` containing the
            turn action.
    """
    def __init__(self, action_space, turn_action_index, *args, **kwargs):
        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)

        self.pi *= 0
        self.pi[turn_action_index] = 1

    def update(self, phi, observation, *args, **kwargs):
        pass

class EpsilonSpin(Policy):
    """Spins except epsilon of the time it randomly choses an action.

    Attributes:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
        turn_action_index (int): Index of ``action_space`` containing the
            turn action.
    """
    def __init__(self,
                 epsilon,
                 action_space,
                 turn_action_index,
                 *args, **kwargs):
        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)

        self.pi *= 0
        self.pi += epsilon / self.pi.size
        self.pi[turn_action_index] += 1 - epsilon

    def update(self, phi, observation, *args, **kwargs):
        pass

class RotateBounce(Policy):
    """Bounces between 0.5 and -0.5 on IMU.

    Attributes:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
    """
    def __init__(self,
                 action_space,
                 *args, **kwargs):
        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)


        self.LEFT = 1
        self.RIGHT = 2
        self.direction = self.LEFT

    def update(self, phi, observation, *args, **kwargs):
        if observation['imu'] > 0.5:
            self.direction = self.RIGHT
        elif observation['imu'] < -0.5:
            self.direction = self.LEFT

        self.pi *= 0
        self.pi[self.direction] += 1

class EpsilonRotateBounce(Policy):
    """Epsilon greedily bounces between 0.5 and -0.5 on IMU.

    Attributes:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
    """
    def __init__(self,
                 epsilon,
                 action_space,
                 *args, **kwargs):
        kwargs['action_space'] = action_space
        Policy.__init__(self, *args, **kwargs)


        self.epsilon = epsilon
        self.LEFT = 1
        self.RIGHT = 2
        self.direction = self.LEFT

    def update(self, phi, observation, *args, **kwargs):
        if observation['imu'] > 0.5:
            self.direction = self.RIGHT
        elif observation['imu'] < -0.5:
            self.direction = self.LEFT

        self.pi *= 0
        self.pi += self.epsilon / self.pi.size
        self.pi[self.direction] += 1 - self.epsilon




class Cumulant:
    """Implements cumulant = next_imu - current_imu
    """
    def __init__(self):
        self.last_imu = 0

    def cumulant(self, obs):
        c = obs['imu'] - self.last_imu
        self.last_imu = obs['imu']
        return c

if __name__ == "__main__":
    try:
        action_manager_process = mp.Process(target=start_action_manager,
                                            name="action_manager",
                                            args=())
        action_manager_process.start()
        
        # robotic parameters
        time_scale = 0.05
        turn_speed = 0.5

        # all available actions
        action_space = np.array([Twist(Vector3(0, 0, 0),
                                       Vector3(0, 0, 0)),
                                 Twist(Vector3(0, 0, 0),
                                       Vector3(0, 0, turn_speed)),
                                 Twist(Vector3(0, 0, 0),
                                       Vector3(0, 0, -turn_speed))])

        # either cycles through hyperparameter possibilities or 
        # runs wall demo once with default hyperparameters
        hps = {'alpha0': 0.05, "lmbda":0.9}

        # learning parameters
        alpha0 = hps['alpha0']
        lmbda = hps['lmbda']

        features_to_use = ['bias', 'imu']
        feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        num_active_features = sum(StateConstants.num_active_features[f] for f in features_to_use)
        num_features = feature_indices.size

        gamma = lambda obs: 0 if abs(obs['imu']) < 0.1 else 1
        cumulant = lambda obs: 1
        test_hp = {'alpha': alpha0 / num_active_features,
                   'beta': 0.001 * alpha0 / num_active_features,
                   'lmbda': lmbda,
                   'alpha0': alpha0,
                   'num_features': num_features,
                   'feature_indices': feature_indices,
                  }

        test_policy = RotateBounce(action_space=action_space,
                                   turn_action_index=1)
        test_learner = GTD(**test_hp)

        behavior_policy = EpsilonRotateBounce(action_space=action_space,
                                      turn_action_index=1,
                                      epsilon=0.1)

        # spin_speed = GVF(cumulant      = cumulant,
        #                  gamma         = gamma,
        #                  target_policy = test_policy,
        #                  learner       = test_learner,
        #                  name          = 'SpinSpeed',
        #                  logger        = rospy.loginfo,
        #                  **test_hp)

        cumulant = Cumulant()
        steps_to_0 = GVF(cumulant      = cumulant.cumulant,
                         gamma         = gamma,
                         target_policy = test_policy,
                         learner       = test_learner,
                         name          = 'SpinSpeed',
                         logger        = rospy.loginfo,
                         **test_hp)



        cumulant_counter = 0
        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [steps_to_0],
                                              features_to_use,
                                              behavior_policy,
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