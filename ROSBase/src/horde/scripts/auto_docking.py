import multiprocessing as mp
import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3

from action_manager import start_action_manager
from algorithms import GTD
from algorithms import GreedyGQ
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

def one_if_in_IR_range():
    for data in topic.data:
        if data!=0:
            return 1
    return 0


if __name__ == "__main__":
    try:
        # random.seed(20170612)

        time_scale = 0.5
        forward_speed = 0.2
        turn_speed = 2

        alpha = 0.0001
        beta = alpha / 10

        one_if_bump = lambda observation: int(any(observation['bump'])) if observation is not None else 0
        one_if_ir = lambda observation: int(any(observation['ir'])) if observation is not None else 0

        go_forward = GoForward(speed=forward_speed)
        action_space = 0
        num_features = len(action_space)
        num_features = 5
        wall_demo = GVF(num_features=14403*num_features,
                        alpha=alpha,
                        beta=beta,
                        gamma=one_if_ir,
                        cumulant=one_if_ir,
                        lambda_= lambda observation: 0.95,
                        alg=GreedyGQ,
                        name='auto_docking',
                        logger=rospy.loginfo)

        behavior_policy = ForwardIfClear(wall_demo, 
                                         vel_linear=forward_speed,
                                         vel_angular=turn_speed)

        topics = [
            # "/camera/depth/image",
            # "/camera/depth/points",
            # "/camera/ir/image",
            # "/camera/rgb/image_raw",
            "/camera/rgb/image_rect_color",
            "/mobile_base/sensors/core",
            "/mobile_base/sensors/dock_ir",
            # "/mobile_base/sensors/imu_data",
            ]

        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [wall_demo],
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
        foreground_process.join()
        action_manager_process.join()        
