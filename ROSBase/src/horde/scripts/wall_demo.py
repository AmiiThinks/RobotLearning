import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3

from algorithms import GTD
from gvf import GVF
from learning_foreground import LearningForeground
from policy import Policy

class GoForward(Policy):
    def __init__(self, speed=0.35):
        Policy.__init__(self)
        self.speed = speed

    def __call__(self, state):
        return Twist(Vector3(self.speed, 0, 0), Vector3(0, 0, 0))

class ForwardIfClear(Policy):
    def __init__(self, gvf, vel_linear=0.35, vel_angular=2):
        Policy.__init__(self)
        self.gvf = gvf
        self.vel_linear = vel_linear
        self.vel_angular = vel_angular

    def __call__(self, state):

        if self.gvf.predict(state) > random.random() or sum(state[:3]):
            action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
        else:
            action = Twist(Vector3(self.vel_linear, 0, 0), Vector3(0, 0, 0))

        return action 

class ForwardIfClearTurtle(Policy):
    def __init__(self, gvf, vel_linear=0.35, vel_angular=2):
        Policy.__init__(self)
        self.gvf = gvf
        self.vel_linear = vel_linear
        self.vel_angular = vel_angular
        self.bump = False

    def __call__(self, state):
        if self.bump:
            state[0] = 0
            self.bump = False

        if self.gvf.predict(state) > random.random() or state[0]:
            action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
            self.bump = True
        else:
            action = Twist(Vector3(self.vel_linear, 0, 0), Vector3(0, 0, 0))

        return action 


if __name__ == "__main__":
    try:

        time_scale = 0.1
        forward_speed = 0.2
        turn_speed = 2

        alpha = 0.000001
        beta = alpha / 10

        one_if_bump = lambda state: int(bool(sum(state[:3])))
        wall_demo = GVF(num_features=14403,
                        alpha=alpha,
                        beta=beta,
                        gamma=one_if_bump,
                        cumulant=one_if_bump,
                        policy=GoForward(speed=forward_speed),
                        off_policy=True,
                        alg=GTD,
                        name='WallDemo',
                        logger=rospy.loginfo)

        behavior_policy = ForwardIfClearTurtle(wall_demo, 
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

        foreground = LearningForeground(alpha, 
                                        time_scale,
                                        [wall_demo],
                                        topics,
                                        behavior_policy)
        foreground.run()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))