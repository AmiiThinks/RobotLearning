"""
Author: Michele Albach, David Quail, Parash Rahman, Niko Yasui, June 2017.

Description:
Behavior policy is responsible for returning the desired action to take.
"""
import geometry_msgs.msg as geom_msg
import random

class Policy:

    def __init__(self):
        # good forward speed is 0.35
        # good angular speed is 2
        self.last_action = None

    def __call__(self, state):
        """
        Implements the policy.

        Input:
            state - state object as defined in learning foreground

        Output:
            twist - Twist object
        """

        # build Twist object
        action = geom_msg.Twist()

        # linear velocity
        #action.linear.x = 0.05

        # # Possible setup
        # # 1: Go forward (meters/s)
        # # 2: Go Backwards (meters/s)
        # # 3: Turn right (rad/s)
        # # 4: Turn left (rad/s)

        # if action==1:
        #     move_cmd.linear.x = 0.2
        # if action==2:
        #     move_cmd.linear.x = -0.2
        # if action==3:
        #     move_cmd.angular.z = 0.5
        # if action==4:
        #     move_cmd.angular.z = -0.5

        # save last action
        self.last_action = action

        return action

