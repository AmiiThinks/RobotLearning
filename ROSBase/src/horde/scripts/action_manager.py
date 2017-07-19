from geometry_msgs.msg import Twist, Vector3
import rospy
from tools import topic_format

class ActionManager():

    def __init__(self):
        self.STOP_ACTION = Twist(Vector3(0,0,0), Vector3(0,0,0))

	self.action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
        self.base_state = None
        rospy.Subscriber("/mobile_base/sensors/core",
                         topic_format["/mobile_base/sensors/core"],
                         self.update_base_state)

    def update_base_state(self, val):
        self.base_state = val

    def update_action(self, action_cmd):
        if (action_cmd.linear.x > 0.0001 or action_cmd.linear.y > 0.0001) and self.base_state.bumper:
            self.action = self.STOP_ACTION
        else:
            self.action = action_cmd

    def run(self):
        rospy.init_node('action_manager', anonymous=True)
        rospy.Subscriber('action_cmd', Twist, self.update_action)

        action_publisher = rospy.Publisher('cmd_vel_mux/input/teleop', 
                                            Twist,
                                            queue_size=1)

        action_pub_rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            # log action
            print_action = "linear: {}, angular: {}".format(self.action.linear.x,
                                                            self.action.angular.z)
            rospy.logdebug("Sending action to Turtlebot: {}".format(print_action))

            # send new actions
            action_publisher.publish(self.action)
            action_pub_rate.sleep()

def start_action_manager():
    try:
        action_manager = ActionManager()
        action_manager.run()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))

