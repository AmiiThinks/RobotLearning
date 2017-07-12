from geometry_msgs.msg import Twist, Vector3
import rospy

class ActionManager():

    def __init__(self):
        self.action = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

    def update_action(self, action_cmd):
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

