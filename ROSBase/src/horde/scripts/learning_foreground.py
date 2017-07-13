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

from gvf import GVF
from policy import Policy
from state_representation import StateManager
from tools import timing, topic_format
from visualize_pixels import Visualize


class LearningForeground:

    def __init__(self, 
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
        self.time_scale = time_scale

        # agent info
        self.gvfs = gvfs
        self.behavior_policy = behavior_policy
        self.state_manager = StateManager()

        # currently costs about 0.0275s per timestep
        rospy.loginfo("Creating visualization.")
        self.visualization = Visualize(self.state_manager.chosen_points,
                                       imsizex=640,
                                       imsizey=480)
        rospy.loginfo("Done creatiing visualization.")

        # previous timestep information
        self.last_action = None
        self.last_phi = self.gvfs[0].learner._phi if self.gvfs else None
        self.last_preds = {g:None for g in self.gvfs}
        self.last_observation = None
        self.last_mu = 1

        # Set up publishers
        pub_name = lambda g, lab: '{}/{}'.format(g, lab) if g else lab
        pub = lambda g, lab: rospy.Publisher(pub_name(g, lab), 
                                             std_msg.Float64, 
                                             queue_size=10)
        action_publisher = rospy.Publisher('action_cmd', 
                                           geom_msg.Twist,
                                           queue_size=1)

        self.publishers = {'avg_rupee': pub('', 'avg_rupee'),
                           'avg_ude': pub('', 'avg_ude'),
                           'action': action_publisher}
        labels = ['prediction', 'rupee', 'ude', 'td_error']
        label_pubs = {g:{l:pub(g.name, l) for l in labels} for g in self.gvfs}
        self.publishers.update(label_pubs)

        rospy.loginfo("Done LearningForeground init.")

    def update_gvfs(self, phi_prime, observation):
        for gvf in self.gvfs:
            pred_before = gvf.predict(self.last_phi)
            gvf.update(self.last_action, 
                       phi_prime,
                       observation, 
                       self.last_observation,
                       self.last_mu)

            # log predictions (optional)
            pred_after = str(gvf.predict(self.last_phi))
            rospy.loginfo("GVF prediction before: {}".format(pred_before))
            rospy.loginfo("GVF prediction after: {}".format(pred_after))

            self.last_preds[gvf] = gvf.predict(phi_prime)

        self.publish_predictions_and_errors()

    def publish_predictions_and_errors(self):

        td = {g:g.learner.delta for g in self.gvfs}
        # rupee = [g.rupee() for g in self.gvfs]
        # ude = [g.ude() for g in self.gvfs]

        # avg_rupee = sum(rupee)/len(rupee)
        # avg_ude = sum(ude)/len(ude)

        for g in self.gvfs:
            self.publishers[g]['prediction'].publish(self.last_preds[g])
            # self.publishers[self.gvfs[i]]['rupee'].publish(rupee[i])
            # self.publishers[self.gvfs[i]]['ude'].publish(ude[i])
            self.publishers[g]['td_error'].publish(td[g])

        # self.publishers['avg_rupee'].publish(avg_rupee)
        # self.publishers['avg_ude'].publish(avg_ude)

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
        ir_status = None

        # clear the bumper queue of unused/old observations
        for _ in range(bumper_num_obs - 1):
            self.recent['/mobile_base/sensors/core'].get()

        # get the last bumper information
        if (bumper_num_obs > 0):
            last_bump_raw = self.recent['/mobile_base/sensors/core'].get().bumper
            bumper_status = (1 if BUMPER_RIGHT & last_bump_raw else 0, 
                             1 if BUMPER_LEFT & last_bump_raw else 0,
                             1 if BUMPER_CENTRE & last_bump_raw else 0)

        # get the last ir information
        if (bumper_num_obs > 0):
            # 
            last_ir_raw_near_left = self.recent['/mobile_base/sensors/dock_ir'].get().NEAR_LEFT
            last_ir_raw_near_center = self.recent['/mobile_base/sensors/dock_ir'].get().NEAR_CENTER
            last_ir_raw_near_right = self.recent['/mobile_base/sensors/dock_ir'].get().NEAR_RIGHT
            last_ir_raw_far_left = self.recent['/mobile_base/sensors/dock_ir'].get().FAR_LEFT
            last_ir_raw_far_center = self.recent['/mobile_base/sensors/dock_ir'].get().FAR_CENTER
            last_ir_raw_far_right = self.recent['/mobile_base/sensors/dock_ir'].get().FAR_RIGHT

            ir_status = (1 if last_ir_raw_far_left or last_ir_raw_near_left else 0, 
                             1 if last_ir_raw_far_center or last_ir_raw_near_center else 0, 
                             1 if last_ir_raw_near_right or last_ir_raw_far_right else 0)

        # get the image processed for the state representation
        image_data = None
        
        # clear the image queue of unused/old observations
        for _ in range(image_num_obs - 1):
            self.recent['/camera/rgb/image_rect_color'].get()

        # get the last image information
        if (image_num_obs > 0):

            br = CvBridge()
            image_data = np.asarray(br.imgmsg_to_cv2(self.recent['/camera/rgb/image_rect_color'].get(),
                desired_encoding="passthrough")) 

        phi = self.state_manager.get_state_representation(image_data, bumper_status, 0)

        # update the visualization of the image data
        self.visualization.update_colours(image_data)

        rospy.loginfo(phi)

        observation = self.state_manager.get_observations(bumper_status, ir_status)
        return phi, observation

    def take_action(self, action):
        self.publishers['action'].publish(action)

    def run(self):
        # Keep track of time for when to avoid sleeping
        sleep_time = self.time_scale - 0.0001
        tic = time.time()

        while not rospy.is_shutdown():
            # To avoid the drift of just calling time.sleep()
            while time.time() < tic:
                time.sleep(0.0001)

            # get new state
            phi_prime, observation = self.create_state()
            self.learner.update(state=last_phi,action=action,observation=observation,next_state=phi_prime)

            # take action
            action, mu = self.gvfs[0].learner.behavior_policy(phi_prime, observation)
            self.take_action(action)

            # learn
            # if self.last_observation is not None:
            #     self.update_gvfs(phi_prime, observation)

            self.last_phi = phi_prime if len(phi_prime) else None
            self.last_action = action
            self.last_mu = mu
            self.last_observation = observation

            # reset tic
            tic += sleep_time

def start_learning_foreground(time_scale,
                              GVFs,
                              topics,
                              Policy):
    try:
        foreground = LearningForeground(time_scale,
                                        GVFs,
                                        topics,
                                        Policy)
        foreground.run()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))

if __name__ == '__main__':
    try:
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
