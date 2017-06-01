"""
Author: Niko Yasui, June 1, 2017.

Description:
Behavior policy is responsibility for returning the desired action to take.
"""
import geometry_msgs.msg as geom_msg
import random

class BehaviorPolicy:

    def __init__(self):
        self.last_action = None

    def __call__(self, state):
        """
        Implements the policy.

        Input:
            state - state object as defined in learning foreground

        Output:
            twist - Twist object
        """

        # create Twist object
        action = geom_msg.Twist()

        # linear velocity
        action.linear.x = 0.05

        # angular velocity
        action.angular.z = 0

        self.last_action = action

        return action


