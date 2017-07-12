import multiprocessing as mp
import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3

from action_manager import start_action_manager
from gtd import GTD
from gvf import GVF
from learning_foreground import start_learning_foreground
from policy import Policy

class GoForward(Policy):
    def __init__(self, speed=0.35):
        Policy.__init__(self)
        self.speed = speed

    def __call__(self, phi, observation):
        return Twist(Vector3(self.speed, 0, 0), Vector3(0, 0, 0)), 1

class ForwardIfClear(Policy):
    def __init__(self, gvf, vel_linear=0.35, vel_angular=2):
        Policy.__init__(self)
        self.gvf = gvf
        self.vel_linear = vel_linear
        self.vel_angular = vel_angular

        # where the last action is recorded according
        # to its respective constants
        self.last_action = None
        self.TURN = 0
        self.FORWARD = 1
        self.STOP = 2

    def __call__(self, phi, observation):
        
        # if self.gvf.predict(phi) > random.random() or sum(observation['bump']):
        #     action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
        #     self.last_action = self.TURN
        # else:
            
        if sum(observation['bump']):
            action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
            self.last_action = self.TURN
        else:
            if self.last_action == self.TURN:
                action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
                self.last_action = self.STOP
            else:
                action = Twist(Vector3(self.vel_linear, 0, 0), Vector3(0, 0, 0))
                self.last_action = self.FORWARD
        return action, 1

if __name__ == "__main__":
    try:
        # random.seed(20170612)
        
        time_scale = 0.5
        forward_speed = 0.2
        turn_speed = 2

        parameters = {'alpha': 0.1, 
                      'beta': 0.01,
                      'lambda': 0.9}

        one_if_bump = lambda observation: int(any(observation['bump'])) if observation is not None else 0
        discount_if_bump = lambda observation: 0 if sum(observation["bump"]) else 0.9
        go_forward = GoForward(speed=forward_speed)

        distance_to_bump = GVF(cumulant = one_if_bump,
                               gamma    = discount_if_bump,
                               target_policy = go_forward,
                               num_features = 14400,
                               parameters = parameters,
                               off_policy = True,
                               alg = GTD,
                               name = 'DistanceToBump',
                               logger = rospy.loginfo)

        behavior_policy = ForwardIfClear(distance_to_bump, 
                                         vel_linear=forward_speed,
                                         vel_angular=turn_speed)

        topics = [
            # "/camera/depth/image",
            # "/camera/depth/points",
            # "/camera/ir/image",
            # "/camera/rgb/image_raw",
            "/camera/rgb/image_rect_color",
            "/mobile_base/sensors/core",
            # "/mobile_base/sensors/dock_ir",
            # "/mobile_base/sensors/imu_data",
            ]

        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [distance_to_bump],
                                              topics,
                                              behavior_policy))

        action_manager_process = mp.Process(target=start_action_manager,
                                            name="action_manager",
                                            args=())
        foreground_process.start()
        action_manager_process.start()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
    finally:
        try:
            foreground_process.join()
            action_manager_process.join()  
        except NameError:
            pass    
