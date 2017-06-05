#!/usr/bin/env python

"""
Author: Michele Albach, David Quail, Parash Rahman, Niko Yasui, June 1, 2017.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""

import geometry_msgs.msg as geo_msg
from Queue import Queue
import rospy
import std_msgs.msg as std_msg
import threading
import time

from behavior_policy import BehaviorPolicy
from state_representation import StateManager
from tools import timing, topic_format

class LearningForeground:

    def __init__(self, learning_rate, time_scale, gvfs, topics):
        
        # set up dictionary to share sensor info
        self.recent = {topic:Queue(0) for topic in topics}

        # set up ros
        rospy.init_node('agent', anonymous=True)
        # setup sensor parsers
        for topic in topics:
            rospy.Subscriber(topic, 
                             topic_format[topic],
                             lambda dat: self.recent[topic].put(dat))

        rospy.loginfo("Started sensor threads.")

        # smooth out the actions
        self.t_len = time_scale
        self.q_len = max(int(time_scale / 0.01), 1)

        # agent info
        self.alpha = learning_rate
        self.gvfs = gvfs
        self.behavior_policy = BehaviorPolicy()
        self.state_manager = StateManager()

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
                                           geo_msg.Twist,
                                           queue_size=self.q_len)
        self.publishers = {'avg_rupee': pub('avg', 'rupee'),
                           'avg_ude': pub('avg', 'ude'),
                           'action': action_publisher}
        labels = ['prediction', 'rupee', 'ude']
        label_pubs = {g:{l:pub(g, l) for l in labels} for g in self.gvfs}
        self.publishers.update(label_pubs)

        rospy.loginfo("Done LearningForeground init.")

    def update_gvfs(self, new_state):
        for gvf in self.gvfs:
            pred_before = gvf.prediction(self.last_state)
            gvf.learn(self.last_state, self.last_action, new_state)

            # log predictions (optional)
            pred_after = str(gvf.prediction(self.last_state))
            rospy.loginfo("GVF prediction before: " + pred_before)
            rospy.loginfo("GVF prediction after: " + pred_after)

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
        rospy.loginfo("Creating state.")
        try:
            # don't create a new variable, this is just for demonstration
            queue = self.recent['mobile_base/sensors/core']

            # get queue size from ALL sensors before reading any of them
            num_recent_obs = queue.qsize()
            obs = []

            # read the sensors
            for _ in num_recent_obs:
                # do whatever parsing you need to do here
                obs =  self.recent['/mobile_base/sensors/core'].get().bumper

            return obs

        except KeyError:
            return False

    def take_action(self, action):
        rospy.loginfo("Sending action to Turtlebot.")
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
            self.take_action(self.behavior_policy(new_state))

            # decide if learning should happen
            if self.last_state is not None and self.gvfs:
                # learn
                self.update_gvfs(new_state)

                # publish predictions and errors
                self.publish_predictions_and_errors(self.last_state)

            self.last_state = new_state

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
        foreground = LearningForeground(learning_rate, time_scale, [], topics)
        foreground.run()

    except rospy.ROSInterruptException:
        pass

    