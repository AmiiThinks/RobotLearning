"""Module to send actions to the turtlebot.

Authors:
    Shibhansh Dohare, Banafsheh Rafiee, Parash Rahman, Niko Yasui.
"""

import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool

from tools import topic_format


class ActionManager():
    """Class that communicates directly with the turtlebot.

    Attributes:
        action (action): The action to send to the turtlebot.
        base_state: The current reading from the turtlebot's 'core'
            topic. Contains bump information.
        termination_flag (bool): Quit if this is true.
        pause_flag (bool): Stop sending actions if this is true.
        stop_once (bool): Send one stop action and resume sending
            ``action``.
    """

    def __init__(self):
        self.STOP_ACTION = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

        self.action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        self.base_state = None
        rospy.Subscriber("/mobile_base/sensors/core",
                         topic_format["/mobile_base/sensors/core"],
                         self.update_base_state)
        self.termination_flag = False
        self.pause_flag = False
        self.stop_once = False

    def update_base_state(self, val):
        """Set ``action`` to ``STOP_ACTION`` if the bumper is on."""
        self.base_state = val
        if self.action.linear.x and self.base_state.bumper:
            self.action = self.STOP_ACTION

    def set_termination_flag(self, termination_flag):
        self.termination_flag = termination_flag

    def set_pause_flag(self, pause_flag):
        self.pause_flag = pause_flag.data

    def update_action(self, action_cmd):
        """Don't go forward if bumping and wait before going forward 
        after turning.
        """
        if action_cmd.linear.x and self.base_state.bumper:
            self.action = self.STOP_ACTION
        elif action_cmd.linear.x and self.action.angular.z:
            self.stop_once = True
            self.action = action_cmd
        else:
            self.action = action_cmd

    def run(self):
        """Send an action at a 40Hz cycle."""
        rospy.init_node('action_manager', anonymous=True)
        rospy.Subscriber('action_cmd', Twist, self.update_action)
        rospy.Subscriber('termination', Bool, self.set_termination_flag)
        rospy.Subscriber('pause', Bool, self.set_pause_flag)

        action_publisher = rospy.Publisher('cmd_vel_mux/input/teleop',
                                           Twist,
                                           queue_size=1)

        action_pub_rate = rospy.Rate(40)

        while not rospy.is_shutdown():
            if self.termination_flag:
                break
            if self.pause_flag is False:
                # log action
                speeds = (self.action.linear.x, self.action.angular.z)
                actn = "linear: {}, angular: {}".format(*speeds)
                rospy.logdebug("Sending action to Turtlebot: {}".format(actn))

                # send new actions
                if self.stop_once:
                    action_publisher.publish(self.STOP_ACTION)
                    self.stop_once = False
                else:
                    action_publisher.publish(self.action)
            action_pub_rate.sleep()


def start_action_manager():
    """Runs the action manager"""
    try:
        action_manager = ActionManager()
        action_manager.run()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
