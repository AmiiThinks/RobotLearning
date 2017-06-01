#!/usr/bin/env python

"""
Author: David Quail, Niko Yasui, June 1, 2017.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""

import geometry_msgs.msg as geo_msg
import rospy
import std_msgs.msg as std_msg
import threading
import time

from behavior_policy import BehaviorPolicy
from observation_manager import ObservationManager

class LearningForeground:

    def __init__(self, learning_rate, time_scale, gvfs, topics):
        
        # set up dictionary to share sensor info
        self.most_recent_obs = dict()
        
        # set up observation manager
        rospy.init_node('observation_manager', anonymous=True)
        t = threading.Thread(target=ObservationManager, 
                             args=(topics, self.most_recent_obs))
        t.daemon = True
        t.start()

        rospy.loginfo("Started ObservationManager thread.")

        # smooth out the actions
        self.action_thread = None
        self.t_len = time_scale
        self.q_len = max(int(time_scale / 0.01), 1)

        # agent info
        self.alpha = learning_rate
        self.gvfs = gvfs
        self.behavior_policy = BehaviorPolicy()

        # previous timestep information
        self.last_action = None
        self.last_state = False
        self.last_preds = {g:None for g in self.gvfs}

        # Set up publishers
        pub_name = lambda g, lab: 'horde_verifier/{}_{}'.format(g, lab)
        pub = lambda g, lab: rospy.Publisher(pub_name(g, lab), 
                                             std_msg.Float64, 
                                             queue_size=10)
        action_publisher = rospy.Publisher('/cmd_vel_mux/input/teleop', 
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
            self.last_preds[gvf] = gvf.prediction(self.last_state)
            gvf.learn(self.last_state, self.last_action, new_state)

            # log predictions (optional)
            pred_before = str(self.last_preds[gvf])
            pred_after = str(gvf.prediction(self.last_state))
            rospy.loginfo("GVF prediction before: " + pred_before)
            rospy.loginfo("GVF prediction after: " + pred_after)


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
        # Do something
        try:
            my_state = self.most_recent_obs['bump_right']
        except KeyError:
            my_state = False
        return my_state

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

            # reset action thread
            if self.action_thread:
                self.action_thread.join()

            # take action
            action = self.behavior_policy(new_state)
            self.action_thread = threading.Thread(target=self.take_action,
                                                  args=[action])
            self.action_thread.start()

            if self.last_state:
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
            # "/camera/rgb/image_rect_color",
            "/mobile_base/sensors/core",
            "/mobile_base/sensors/dock_ir",
            "/mobile_base/sensors/imu_data",
            ]
        foreground = LearningForeground(learning_rate, time_scale, [], topics)
        foreground.run()

    except rospy.ROSInterruptException:
        pass

    