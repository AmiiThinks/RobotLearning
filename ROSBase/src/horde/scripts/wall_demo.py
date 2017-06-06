import random
import rospy
import geometry_msgs.msg as geom_msg

from gvf import GVF
from learning_foreground import LearningForeground
from policy import Policy

class GoForward(Policy):
    def __init__(self, speed=0.35):
        Policy.__init__(self)
        self.speed = speed

    def __call__(self, state):
        action = geom_msg.Twist()
        action.linear.x = self.speed
        return action

class ForwardIfClear(Policy):
    def __init__(self, gvf, vel_linear=0.35, vel_angular=2):
        Policy.__init__(self)
        self.gvf = gvf
        self.vel_linear = vel_linear
        self.vel_angular = vel_angular

    def __call__(self, state):
        action = geom_msg.Twist()

        if (len(state) and self.gvf.prediction(state) > random.random()) or sum(state[:3]):
            action.angular.z = self.vel_angular
        else:
            action.linear.x = self.vel_linear
        return action 

if __name__ == "__main__":
    try:
        learning_rate = 0.5
        time_scale = 0.5

        one_if_bump = lambda state: bool(sum(state[:3]))      
        wall_demo = GVF(n_features=14403,
                        alpha=learning_rate,
                        isOffPolicy=True,
                        name='WallDemo')
        wall_demo.gamma = one_if_bump
        wall_demo.cumulant = one_if_bump
        wall_demo.policy = GoForward(speed=0.05)

        behavior_policy = ForwardIfClear(wall_demo)

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