import random
import rospy
from geometry_msgs.msg import Twist, Vector3

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

        if self.gvf.prediction(state) > random.random() or sum(state[:3]):
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

        if self.gvf.prediction(state) > random.random() or state[0]:
            action = Twist(Vector3(0, 0, 0), Vector3(0, 0, self.vel_angular))
            self.bump = True
        else:
            action = Twist(Vector3(self.vel_linear, 0, 0), Vector3(0, 0, 0))

        return action 


if __name__ == "__main__":
    try:

        learning_rate = 0.000001
        time_scale = 0.1
        speed = 3
        turn = 2

        one_if_bump = lambda state: int(bool(sum(state[:3])))
        wall_demo = GVF(n_features=14403,
                        alpha=learning_rate,
                        isOffPolicy=True,
                        name='WallDemo',
                        learner='GTD')
        wall_demo.gamma = one_if_bump
        wall_demo.cumulant = one_if_bump
        wall_demo.policy = GoForward(speed=speed)

        behavior_policy = ForwardIfClearTurtle(wall_demo, 
                                               vel_linear=speed,
                                               vel_angular=turn)

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

        foreground = LearningForeground(learning_rate, 
                                        time_scale,
                                        [wall_demo],
                                        topics,
                                        behavior_policy)
        foreground.run()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))