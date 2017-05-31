#!/usr/bin/env python

import rospy
import math
import time
from geometry_msgs.msg import Twist

def policy():
   # behaviour policy: goes straight unless it hits a wall or predicts it
   # all commented because there are too many unknowns to have it run yet
    
#    while # ???? (not sure what to put here)
 #       if # bumper is triggered
  #          # back up then turn right
   #         take_action(2,0.05)
    #        take_action(3,math.pi/2)
#        else if # prediction is high enough
 #           # turn right
  #          take_action(3,math.pi/2)
   #     else:
    #        # go forward
     #       take_action(1,0.05)
            

class takeAction():

    # 1: Go forward
    # 2: Go Backwards
    # 3: Turn right
    # 4: Turn left
    # if action is 1 or 2, distance is in meters
    # if action is 3 or 4, distance is in radians

    def __init__(self,action,distance):
        
        rospy.init_node('action', anonymous=False)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
    
        r = rospy.Rate(10);
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0

        # Set appropriate linear or angular speed
        # Set time to move it by dividing distance by speed
        if action==1:
            move_cmd.linear.x = 0.2
            t = distance / 0.2
        if action==2:
            move_cmd.linear.x = -0.2
            t = distance / 0.2
        if action==3:
            move_cmd.angular.z = 0.5
            t = distance / 0.5
        if action==4:
            move_cmd.angular.z = -0.5
            t = distance / 0.5

        # Perform loop for the amount of time to move that distance
        start = time.time()
        while((time.time() - start) < t):
            self.cmd_vel.publish(move_cmd)
            r.sleep()

if __name__ == '__main__':
    # Turn 180 degrees to the left
    takeAction(4,math.pi)


# NOTE: I tried this and it doesn't work. It moves properly but less than
# the distance that it's supposed to. So depending on how specific we
# need to be this might not work. I'll continue looking for other methods.
