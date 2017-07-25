from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool
import rospy
from tools import topic_format

class ActionManager():

    def __init__(self):
        self.STOP_ACTION = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

        self.action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        self.base_state = None
        rospy.Subscriber("/mobile_base/sensors/core",
                         topic_format["/mobile_base/sensors/core"],
                         self.update_base_state)
        self.termination_flag = False
    def update_base_state(self, val):
        self.base_state = val

    def set_termination_flag(self, termination_flag):
        self.termination_flag = termination_flag

    def set_pause_flag(self, pause_flag):
        self.pause_flag = pause_flag

    def update_action(self, action_cmd):
        if action_cmd.linear.x and self.base_state.bumper:
            self.action = self.STOP_ACTION
        else:
            self.action = action_cmd

    def run(self):
        rospy.init_node('action_manager', anonymous=True)
        rospy.Subscriber('action_cmd', Twist, self.update_action)
        rospy.Subscriber('termination', Bool, self.set_termination_flag)
        rospy.Subscriber('pause', Bool, self.set_pause_flag)

        action_publisher = rospy.Publisher('cmd_vel_mux/input/teleop', 
                                           Twist,
                                           queue_size=1)

        action_pub_rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            if self.termination_flag:
                break
            # log action
            speeds = (self.action.linear.x, self.action.linear.z)
            actn = "linear: {}, angular: {}".format(*speeds)
            rospy.logdebug("Sending action to Turtlebot: {}".format(actn))

            # send new actions
            if self.pause_flag is False:
                action_publisher.publish(self.action)
            action_pub_rate.sleep()

def start_action_manager():
    try:
        action_manager = ActionManager()
        action_manager.run()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))