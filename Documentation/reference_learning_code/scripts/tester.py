import rospy
import time

from std_msgs.msg import Float64

from dynamixel_driver.dynamixel_const import *


def performAction(action):
    # Take the action and issue the actual dynamixel command
    pub = rospy.Publisher('tilt_controller/command', Float64, queue_size=10)
    print("Performning action: " + str(action))
    if (action == 1):
        # Move left
        pub.publish(0.0)
    elif (action == 2):
        pub.publish(3.0)
    elif (action == 0):
        pubSpeed = rospy.Publisher('tilt_controller/set_speed', Float64, queue_size=10)
        pubSpeed.publish(0.0)

def test():
    rospy.init_node('tester', anonymous=True)
    performAction(1)
    time.sleep(2)
    performAction(2)
    time.sleep(0.1)

def testInterupt():
    rospy.init_node('tester', anonymous=True)
    performAction(1)
    time.sleep(2)
    performAction(2)
    time.sleep(0.1)
    performAction(0)
