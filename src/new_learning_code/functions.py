#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

def policy():

    rospy.init_node('policy', anonymous=False)
    self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

#    while # ???? (not sure what to put here)
 #       if # bumper is triggered
  #          # back up then turn right
   #         take_action(2)
    #        take_action(3)
#        else if # prediction is high enough
 #           # turn right
  #          take_action(3)
   #     else:
    #        # go forward
     #       take_action(1)
            

def take_action(action):

    # 1: Go forward
    # 2: Go Backwards
    # 3: Turn right
    # 4: Turn left
    
    rospy.init_node('action', anonymous=False)
    self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

    r = rospy.Rate(10);
    move_cmd = Twist()
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0

    if action==1:
        move_cmd.linear.x = 0.2
    if action==2:
        move_cmd.linear.x = -0.2
    if action==3:
        move_cmd.angular.z = 0.2
    if action==4:
        move_cmd.angular.z = -0.2

    for i in range(20): # For ~two seconds
        self.cmd_vel.publish(move_cmd)
        r.sleep()


if __name__ == '__main__':
    try:
        take_action(1)
    except:
        rospy.loginfo("action node terminated.")

        
