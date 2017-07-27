from __future__ import division
import numpy as np
import random
import rospy
import sys
from geometry_msgs.msg import Twist, Vector3

import multiprocessing as mp
from gvf import GVF
from gtd import GTD
from return_calculator import start_return_calculator
from action_manager import start_action_manager
from state_representation import StateConstants
from policy import Policy


# go forward with probability 0.9 and left with probability 0.1
class GoForwardWithRandomness(Policy):
    def __init__(self, forward_percentage, *args, **kwargs):

        self.forward_percentage = forward_percentage

        # where the last action is recorded according
        # to its respective constants
        self.TURN = 2
        self.FORWARD = 1
        self.STOP = 0

        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observations, *args, **kwargs):
        
        self.pi = np.zeros(self.action_space.size)

        if bool(sum(observations["bump"])):
            self.pi[self.TURN] = 1
        else:
            if self.last_index == self.TURN:
                self.pi[self.STOP] = 1
            else:
                self.pi[self.FORWARD] = self.forward_percentage
                self.pi[self.TURN] = 1 - self.forward_percentage

# go forward if bump sensor is off
class GoForwardIfNotBump(Policy):
    def __init__(self, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        self.TURN = 2
        self.FORWARD = 1
        self.STOP = 0

        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        if bool(sum(observations["bump"])):
            chosen_index = self.STOP
        else:
            chosen_index = self.FORWARD

        self.pi = np.zeros(self.action_space.size)
        self.pi[chosen_index] = 1

if __name__ == "__main__":
    try:

        time_scale = 0.5
        forward_speed = 0.15
        turn_speed = 1.5

        parameters = {'alpha': 0,
                      'alpha0': 0,
                      'lmbda': 0,
                     }
        features_to_use = ['image','bump','bias']
        feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        num_features = feature_indices.size
        
        one_if_bump = lambda observations: int(bool(sum(observations["bump"])))
        discount_if_bump = lambda observations: 0 if bool(sum(observations["bump"])) else 0.9

        # all available actions
        action_space = np.array([Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)),
                         Twist(Vector3(forward_speed,0,0), Vector3(0,0,0)),
                         Twist(Vector3(0, 0, 0), Vector3(0, 0, turn_speed))])

        distance_to_bump = GVF(cumulant = one_if_bump,
                               gamma    = discount_if_bump,
                               target_policy = None,
                               num_features = num_features,
                               learner = None,
                               name = 'DistanceToBump',
                               logger = rospy.loginfo,
                               feature_indices = feature_indices,
                               **parameters)


        behavior_policy = GoForwardWithRandomness(forward_percentage=0.9,
                                            action_space=action_space,
                                            feature_indices=feature_indices)
        target_policy   = GoForwardIfNotBump(action_space=action_space,
                                             feature_indices=feature_indices)

        # Run action_manager     
        action_manager_process = mp.Process(target=start_action_manager,
                                            name="action_manager",
                                            args=())


        action_manager_process.start()

        # Run learning_foregound     
        return_calculator_process = mp.Process(target=start_return_calculator,
                                                 name="return_calculator",
                                                 args=(time_scale,
                                                      distance_to_bump,
                                                      num_features,
                                                      features_to_use,
                                                      behavior_policy,
                                                      target_policy))
        return_calculator_process.start()
        


    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
    finally:
        try:
            return_calculator_process.join()
            action_manager_process.join()  
        except NameError:
            pass    
        