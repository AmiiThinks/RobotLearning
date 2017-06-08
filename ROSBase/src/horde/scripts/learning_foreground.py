#!/usr/bin/env python

"""
Author: Michele Albach, David Quail, Parash Rahman, Niko Yasui, June 1, 2017.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""

from cv_bridge.core import CvBridge
import numpy as np
import geometry_msgs.msg as geom_msg
from Queue import Queue
import rospy
import std_msgs.msg as std_msg
import threading
import time

from policy import Policy
from gvf import GVF
from state_representation import StateManager
from tools import timing, topic_format
from visualize_pixels import Visualize


class LearningForeground:

    def __init__(self, 
                 learning_rate, 
                 time_scale, 
                 gvfs, 
                 topics, 
                 behavior_policy):
        
        # set up dictionary to share sensor info
        self.recent = {topic:Queue(0) for topic in topics}

        # set up ros
        rospy.init_node('agent', anonymous=True)
        # setup sensor parsers
        for topic in topics:
            rospy.Subscriber(topic, 
                             topic_format[topic],
                             self.recent[topic].put)

        rospy.loginfo("Started sensor threads.")

        # smooth out the actions
        self.t_len = time_scale
        self.q_len = max(int(time_scale / 0.1), 1)

        # agent info
        self.alpha = learning_rate
        self.gvfs = gvfs
        self.behavior_policy = behavior_policy
        self.state_manager = StateManager()

        # set up voronoi for image visualization
        # note: this takes a while but only has to be done once
        self.visualization = Visualize(self.state_manager.chosen_points,1080,1080) # Should give size of images received from robot

        # previous timestep information
        self.last_action = None
        self.last_state = None
        self.last_preds = {g:None for g in self.gvfs}

        # Set up publishers
        pub_name = lambda g, lab: 'horde_verifier/{}_{}'.format(g, lab)
        pub = lambda g, lab: rospy.Publisher(pub_name(g, lab), 
                                             std_msg.Float64, 
                                             queue_size=10)
        action_publisher = rospy.Publisher('cmd_vel_mux/input/teleop', 
                                           geom_msg.Twist,
                                           queue_size=self.q_len)

        self.publishers = {'avg_rupee': pub('avg', 'rupee'),
                           'avg_ude': pub('avg', 'ude'),
                           'action': action_publisher}
        labels = ['prediction', 'rupee', 'ude']
        label_pubs = {g:{l:pub(g.name, l) for l in labels} for g in self.gvfs}
        self.publishers.update(label_pubs)

        rospy.loginfo("Done LearningForeground init.")

    def update_gvfs(self, new_state):
        for gvf in self.gvfs:
            pred_before = gvf.prediction(self.last_state)
            gvf.learn(self.last_state, self.last_action, new_state)

            # log predictions (optional)
            pred_after = str(gvf.prediction(self.last_state))
            rospy.loginfo("GVF prediction before: {}".format(pred_before))
            rospy.loginfo("GVF prediction after: {}".format(pred_after))

            self.last_preds[gvf] = gvf.prediction(new_state)


    def publish_predictions_and_errors(self, state):

        preds = [g.prediction(state) for g in self.gvfs]
        rupee = [g.rupee() for g in self.gvfs]
        ude = [g.ude() for g in self.gvfs]

        avg_rupee = sum(rupee)/len(rupee)
        avg_ude = sum(ude)/len(ude)

        for i in range(len(self.gvfs)):
            self.publishers[self.gvfs[i]]['prediction'].publish(preds[i])
            self.publishers[self.gvfs[i]]['rupee'].publish(rupee[i])
            self.publishers[self.gvfs[i]]['ude'].publish(ude[i])

        self.publishers['avg_rupee'].publish(avg_rupee)
        self.publishers['avg_ude'].publish(avg_ude)

    def create_state(self):
        # TODO: consider moving the data processing elsewhere

        rospy.loginfo("Creating state...")
        # get queue size from ALL sensors before reading any of them
        bumper_num_obs = self.recent['/mobile_base/sensors/core'].qsize()
        image_num_obs = self.recent['/camera/rgb/image_rect_color'].qsize()

        # bumper constants from http://docs.ros.org/hydro/api/kobuki_msgs/html/msg/SensorState.html
        BUMPER_RIGHT  = 1
        BUMPER_CENTRE = 2
        BUMPER_LEFT   = 4

        # variables that will be passed to the state manager to create the state
        bumper_status = None

        # clear the bumper queue of unused/old observations
        for _ in range(bumper_num_obs - 1):
            self.recent['/mobile_base/sensors/core'].get()

        # get the last bumper information
        if (bumper_num_obs > 0):
            last_bump_raw = self.recent['/mobile_base/sensors/core'].get().bumper
            bumper_status = (BUMPER_RIGHT & last_bump_raw, 
                             BUMPER_LEFT & last_bump_raw, 
                             BUMPER_CENTRE & last_bump_raw)
               


        # get the image processed for the state representation
        image_data = None

        # update the visualization of the image data
        self.visualization.update_colours(image_data)
        
        # clear the image queue of unused/old observations
        for _ in range(image_num_obs - 1):
            self.recent['/camera/rgb/image_rect_color'].get()

        # get the last image information
        if (image_num_obs > 0):

            br = CvBridge()
            image_data = np.asarray(br.imgmsg_to_cv2(self.recent['/camera/rgb/image_rect_color'].get(),
                desired_encoding="passthrough")) 

        state_rep = self.state_manager.get_state_representation(image_data, bumper_status, 0)

        rospy.loginfo(state_rep)

        return state_rep

    def take_action(self, action):

        # log action
        print_action = "linear: {}, angular: {}".format(action.linear.x,
                                                        action.angular.z)
        rospy.loginfo("Sending action to Turtlebot: {}".format(print_action))

        # send new actions
        [self.publishers['action'].publish(action) for _ in range(self.q_len)]

    def run(self):
        # Keep track of time for when to avoid sleeping
        sleep_time = self.t_len - 0.0001
        tic = time.time()

        while not rospy.is_shutdown():
            # To avoid the drift of just calling time.sleep()
            while time.time() < tic:
                time.sleep(0.0001)

            # get new state
            new_state = self.create_state()

            # take action
            action = self.behavior_policy(new_state)
            self.take_action(action)

            # decide if learning should happen
            if self.last_state is not None and self.gvfs:
                # learn
                self.update_gvfs(new_state)

                # publish predictions and errors
                self.publish_predictions_and_errors(self.last_state)

            self.last_state = new_state if len(new_state) else None
            self.last_action = action

            # reset tic
            tic += sleep_time


if __name__ == '__main__':
    try:
        learning_rate = 0.5
        time_scale = 0.5

        topics = [
            # "/camera/depth/image",
            # "/camera/depth/points",
            # "/camera/ir/image",
            # "/camera/rgb/image_raw",
            "/camera/rgb/image_rect_color",
            "/mobile_base/sensors/core",
            # "/mobile_base/sensors/dock_ir",
            # "/mobile_base/sensors/imu_data",
            ]
        foreground = LearningForeground(learning_rate, 
                                        time_scale,
                                        [],
                                        topics,
                                        Policy())
        foreground.run()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))

    
