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
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import Twist, Vector3
import threading
import time
import sys
import pickle

from gvf import GVF
from state_representation import StateManager
from gentest_state_representation import GenTestStateManager
from tools import timing, topic_format
from visualize_pixels import Visualize


class LearningForeground:

    def __init__(self, 
                 time_scale, 
                 gvfs, 
                 topics, 
                 behavior_policy,
                 control_gvf=None):
        
        # set up dictionary to receive sensor info
        self.recent = {topic:Queue(0) for topic in topics}
        self.control_gvf = control_gvf
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
        self.r = rospy.Rate(1.0/self.time_scale)

        # agent info
        self.gvfs = gvfs
        self.behavior_policy = behavior_policy
        self.avg_td_err = None

        self.state_manager = StateManager()

        # currently costs about 0.0275s per timestep
        rospy.loginfo("Creating visualization.")
        self.visualization = Visualize(self.state_manager.pixel_mask,
                                       imsizex=640,
                                       imsizey=480)
        rospy.loginfo("Done creatiing visualization.")

        # previous timestep information
        self.last_action = None
        self.last_phi = None
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

        self.publishers = {'action': action_publisher}
        labels = ['prediction', 'td_error', 'avg_td_error', 'rupee', 
                  'cumulant']
        label_pubs = {g:{l:pub(g.name, l) for l in labels} for g in self.gvfs}
        self.publishers.update(label_pubs)

        rospy.loginfo("Done LearningForeground init.")

    @timing
    def update_gvfs(self, phi_prime, observation):
        for gvf in self.gvfs:
            gvf.update(self.last_observation,
                       self.last_phi,
                       self.last_action, 
                       observation,
                       phi_prime, 
                       self.last_mu)

        # publishing
        for gvf in self.gvfs:
            self.publishers[gvf]['prediction'].publish(self.last_preds[gvf])
            self.publishers[gvf]['cumulant'].publish(gvf.cumulant_t)
            self.publishers[gvf]['td_error'].publish(gvf.td_error)
            self.publishers[gvf]['avg_td_error'].publish(gvf.avg_td_error)
            self.publishers[gvf]['rupee'].publish(gvf.rupee())

    def create_state(self):
        # TODO: consider moving the data processing elsewhere

        rospy.loginfo("Creating state...")
        # get queue size from ALL sensors before reading any of them
        bumper_num_obs = self.recent['/mobile_base/sensors/core'].qsize()
        ir_num_obs = self.recent['/mobile_base/sensors/dock_ir'].qsize()
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
        for _ in range(ir_num_obs - 1):
            self.recent['/mobile_base/sensors/dock_ir'].get()
        if (ir_num_obs > 0):
            # 
            last_ir_raw_left = ord(self.recent['/mobile_base/sensors/dock_ir'].get().data[0])
            last_ir_raw_center = ord(self.recent['/mobile_base/sensors/dock_ir'].get().data[1])
            last_ir_raw_right = ord(self.recent['/mobile_base/sensors/dock_ir'].get().data[2])

            ir_status = (last_ir_raw_left, last_ir_raw_center, last_ir_raw_right)
        # get the image processed for the state representation
        image_data = None
        
        # clear the image queue of unused/old observations
        for _ in range(image_num_obs - 1):
            self.recent['/camera/rgb/image_rect_color'].get()

        # get the last image information
        if (image_num_obs > 0):

            br = CvBridge()
            image_data = self.recent['/camera/rgb/image_rect_color'].get()
            image_data = br.imgmsg_to_cv2(image_data)
            image_data = np.asarray(image_data)

        primary_gvf_weight = None
        if len(self.gvfs) > 0:
            primary_gvf_weight = self.gvfs[0].learner.theta
        phi = self.state_manager.get_phi(image_data, bumper_status, primary_gvf_weight)

        # update the visualization of the image data
        self.visualization.update_colours(image_data)

        # takes a long time, only uncomment if necessary
        # rospy.loginfo(phi)

        observation = self.state_manager.get_observations(bumper_status, ir_status)
        return phi, observation

    def take_action(self, action):
        self.publishers['action'].publish(action)

    def reset_episode(self):
        for i in range(10):
            action, mu = self.gvfs[0].learner.take_random_action()
            self.take_action(action)
            rospy.loginfo('taking random action number: {}'.format(i))
            # with open('/home/turtlebot/average_rewards','w') as f:
            #     pickle.dump(average_rewards, f)
            self.r.sleep()
    
    def run(self):
        # Keep track of time for when to avoid sleeping

        finished_episode = False

        while not rospy.is_shutdown():

            # get new state
            phi_prime, observation = self.create_state()

            # make prediction
            self.last_preds = {g:g.predict(phi_prime) for g in self.gvfs}

            # take action
            action, mu = self.behavior_policy(phi_prime,observation)
            self.take_action(action)

            # learn
            if self.last_observation is not None:
                self.update_gvfs(phi_prime, observation)

            if self.control_gvf != None:
                finished_episode = self.control_gvf.cumulant(observation) == 1

            if finished_episode:
                self.reset_episode()

            self.last_phi = phi_prime if len(phi_prime) else None
            self.last_action = action
            self.last_mu = mu
            self.last_observation = observation

            # sleep until next time step
            self.r.sleep()

def start_learning_foreground(time_scale,
                              GVFs,
                              topics,
                              policy,
                              control_gvf=None):

    try:
        foreground = LearningForeground(time_scale,
                                        GVFs,
                                        topics,
                                        policy,
                                        control_gvf)

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
                                        policy)
        foreground.run()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
