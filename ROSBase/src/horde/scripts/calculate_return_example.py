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


# go forward with probability 0.9 and left with probability 0.1
class GoForwardWithRandomness():
    def __init__(self, vel_linear=0.35, vel_angular=2):
        self.vel_linear = vel_linear
        self.vel_angular = vel_angular

        # where the last action is recorded according
        # to its respective constants
        self.last_action = None
        self.TURN = 0
        self.FORWARD = 1
        self.STOP = 2

        self.forward_percentage = 0.9

    def __call__(self, phi, observations):
        
        if bool(sum(observations["bump"])):
            action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
            mu = 1
            self.last_action = self.TURN
        else:
            if self.last_action == self.TURN:
                action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
                mu = 1
                self.last_action = self.STOP
            else:
                if np.random.random() < self.forward_percentage:
                    action = Twist(Vector3(self.vel_linear, 0, 0), Vector3(0, 0, 0))
                    mu = self.forward_percentage
                    self.last_action = self.FORWARD
                else:
                    action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
                    mu = 1 - self.forward_percentage
                    self.last_action = self.TURN
        
        return action, mu

# go forward if bump sensor is off
class GoForwardIfNotBump():
    def __init__(self, vel_linear=0.35, vel_angular=2):
        self.vel_linear = vel_linear
        self.vel_angular = vel_angular

        # where the last action is recorded according
        # to its respective constants
        self.TURN = 0
        self.FORWARD = 1
        self.STOP = 2

    def __call__(self, phi, observations):
        pi = None
        if bool(sum(observations["bump"])):
            action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        else:
            action = Twist(Vector3(self.vel_linear, 0, 0), Vector3(0, 0, 0))
        return action, pi

if __name__ == "__main__":
    try:

        time_scale = 0.5
        forward_speed = 0.15
        turn_speed = 1.5

        parameters = dict()
        features_to_use = ['image','bump','bias']
        num_features = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use]).size
        
        one_if_bump = lambda observations: int(bool(sum(observations["bump"])))
        discount_if_bump = lambda observations: 0 if bool(sum(observations["bump"])) else 0.9

        
        distance_to_bump = GVF(cumulant = one_if_bump,
                               gamma    = discount_if_bump,
                               target_policy = None,
                               num_features = num_features,
                               parameters = parameters,
                               learner = None,
                               name = 'DistanceToBump',
                               logger = rospy.loginfo,
                               features_to_use = features_to_use)


        behavior_policy = GoForwardWithRandomness(vel_linear=forward_speed,
                                                  vel_angular=turn_speed)
        target_policy   = GoForwardIfNotBump(vel_linear=forward_speed,
                                             vel_angular=turn_speed)

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
        