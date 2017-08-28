#!/usr/bin/env python
"""Interupting the existing auto-docking routine in kobuki.

This module is taken from the 'client' of existing auto-docking algorithm - 
https://github.com/yujinrobot/kobuki/blob/devel/kobuki_auto_docking/scripts/DockDriveActionClient.py.

The modifications are done to control and modify the existing auto-docking algorithm.

Current modifications preempt the goal once the robot has aligned itself with the docking station.

For any further clearifications contact - shibhanshdohare1997@gmail.com
"""

import roslib; roslib.load_manifest('kobuki_auto_docking')
import rospy
import sys, os

import actionlib
from kobuki_msgs.msg import AutoDockingAction, AutoDockingGoal
from actionlib_msgs.msg import GoalStatus

def doneCb(status, result):
  if 0: print ''
  elif status == GoalStatus.PENDING   : state='PENDING'
  elif status == GoalStatus.ACTIVE    : state='ACTIVE'
  elif status == GoalStatus.PREEMPTED : state='PREEMPTED'
  elif status == GoalStatus.SUCCEEDED : state='SUCCEEDED'
  elif status == GoalStatus.ABORTED   : state='ABORTED'
  elif status == GoalStatus.REJECTED  : state='REJECTED'
  elif status == GoalStatus.PREEMPTING: state='PREEMPTING'
  elif status == GoalStatus.RECALLING : state='RECALLING'
  elif status == GoalStatus.RECALLED  : state='RECALLED'
  elif status == GoalStatus.LOST      : state='LOST'
  # Print state of action server
  print 'Result - [ActionServer: ' + state + ']: ' + result.text

def activeCb():
  if 0: print 'Action server went active.'

def feedbackCb(feedback):
  """Gets the feedback from the auto-docking server.

  Gets the feedback from the server and preempts when aligned

  Can be changed to preempt at any step of the algorithm.

  """
  print 'Feedback: [DockDrive: ' + feedback.state + ']: ' + feedback.text
  global client
  global count
  count += 1
  if count > 25:
    if str(feedback.state).startswith('ALIGNED'):
        client.cancel_goal()
        client.cancel_goal()

def dock_drive_client():
  # add timeout setting
  global client
  global count
  count = 0
  client = actionlib.SimpleActionClient('dock_drive_action', AutoDockingAction)
  while not client.wait_for_server(rospy.Duration(5.0)):
    if rospy.is_shutdown(): return
    print 'Action server is not connected yet. still waiting...'

  goal = AutoDockingGoal();
  client.send_goal(goal, doneCb, activeCb, feedbackCb)
  print 'Goal: Sent.'
  rospy.on_shutdown(client.cancel_goal)
  client.wait_for_result()

  #print '    - status:', client.get_goal_status_text()
  return client.get_result()

if __name__ == '__main__':
  try:
    rospy.init_node('dock_drive_client_py', anonymous=True)
    dock_drive_client()
    #print ''
    #print "Result: ", result
  except rospy.ROSInterruptException: 
    print "program interrupted before completion"
