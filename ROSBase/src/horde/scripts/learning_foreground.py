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
import random
import subprocess
import os, sys

from gvf import GVF
from state_representation import StateManager
from gentest_state_representation import GenTestStateManager
import tools
from tools import timing
from visualize_pixels import Visualize

class LearningForeground:
    def __init__(self,
                 time_scale,
                 gvfs,
                 features_to_use,
                 behavior_policy,
                 control_gvf=None):
       
        self.features_to_use = features_to_use + ['core']
        if 'ir' not in features_to_use:
            self.features_to_use = features_to_use + ['ir']
        topics = filter(lambda x: x, 
                        [tools.features[f] for f in self.features_to_use])

        # set up dictionary to receive sensor info
        self.recent = {topic:Queue(0) for topic in topics}

        # set up ros
        rospy.init_node('agent', anonymous=True)

        # setup sensor parsers
        for topic in topics:
            rospy.Subscriber(topic, 
                             tools.topic_format[topic],
                             self.recent[topic].put)
        self.topics = topics

        rospy.loginfo("Started sensor threads.")

        # smooth out the actions
        self.time_scale = time_scale
        self.r = rospy.Rate(1.0/self.time_scale)

        # agent info
        self.gvfs = gvfs
        self.control_gvf = control_gvf
        self.behavior_policy = behavior_policy
        self.avg_td_err = None

        self.state_manager = GenTestStateManager(features_to_use)
        self.img_to_cv2 = CvBridge().compressed_imgmsg_to_cv2

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
        pause_publisher = rospy.Publisher('pause', 
                                                std_msg.Bool,
                                                queue_size=1)

        self.publishers = {'action': action_publisher,'pause': pause_publisher}
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
            self.publishers[gvf]['cumulant'].publish(gvf.last_cumulant)
            self.publishers[gvf]['td_error'].publish(gvf.td_error)
            self.publishers[gvf]['avg_td_error'].publish(gvf.avg_td_error)
            self.publishers[gvf]['rupee'].publish(gvf.rupee())

    @timing
    def create_state(self):
        # TODO: consider moving the data processing elsewhere

        rospy.loginfo("Creating state...")

        # bumper constants from http://docs.ros.org/hydro/api/kobuki_msgs/html/msg/SensorState.html
        bump_codes = [1, 4, 2]
        # BUMPER_RIGHT  = 1
        # BUMPER_CENTRE = 2
        # BUMPER_LEFT   = 4

        # build data to make phi
        data = {k: None for k in tools.features.keys()}
        for source in self.features_to_use:
            temp = None
            try:
                while True:
                    temp = self.recent[tools.features[source]].get_nowait()
            except:
                pass
            data[source] = temp

        if data['core'] is not None:
            bump = data['core'].bumper
            data['bump'] = map(lambda x: bool(x & bump), bump_codes)
            data['charging'] = bool(data['core'].charger & 2)
        else:
            data['bump'] = None
            data['charging'] = None
        if data['ir'] is not None:
            data['ir'] = [ord(obs) for obs in data['ir'].data]
        if data['image'] is not None:
            cv2_image = self.img_to_cv2(data['image'])
            if cv2_image is None:
                data['image'] = None
            else:
                data['image'] = np.asarray(cv2_image)
        if data['odom'] is not None:
            pos = data['odom'].pose.pose.position
            data['odom'] = np.array([pos.x, pos.y])
        if data['imu'] is not None:
            data['imu'] = data['imu'].orientation.z
        if 'bias' in self.features_to_use:
            data['bias'] = True
        data['weights'] = self.gvfs[0].learner.theta if self.gvfs else None

        phi = self.state_manager.get_phi(**data)

        # update the visualization of the image data
        if (data['image'] is not None):
            self.visualization.update_colours(data['image'])

        # takes a long time, only uncomment if necessary
        # rospy.loginfo(phi)

        observation = self.state_manager.get_observations(**data)
        observation['action'] =self.last_action
        return phi, observation

    def take_action(self, action):
        self.publishers['action'].publish(action)

    def reset_episode(self):
        temp = random.randint(0,20)
        for i in range(30):
            if i < 10:
                self.take_action(Twist(Vector3(-0.1, 0, 0), Vector3(0, 0, 0)))
            elif i >= 10 and i < 10+temp:
                self.take_action(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.5)))
            else:
                self.take_action(Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0)))
            rospy.loginfo('taking random action number: {}'.format(i))
            self.r.sleep()
        self.publishers["pause"].publish(True)
        os.system('python interrupt_auto_docking.py')
        self.publishers["pause"].publish(False)

        # for i in range(random.randint(0,40)):
        #     action, mu = self.gvfs[0].learner.take_random_action()
        #     self.take_action(action)
        #     rospy.loginfo('taking random action number: {}'.format(i))
        #     # with open('/home/turtlebot/average_rewards','w') as f:
        #     #     pickle.dump(average_rewards, f)
        #     self.r.sleep()        
    
    def run(self):

        while not rospy.is_shutdown():
            # get new state
            phi_prime, observation = self.create_state()

            # make prediction
            self.last_preds = {g:g.predict(phi_prime) for g in self.gvfs}
    
            # take action
            action, mu = self.behavior_policy(phi_prime, observation)
            self.take_action(action)

            # learn
            if self.last_observation is not None:
                self.update_gvfs(phi_prime, observation)

            # check if episode is over
            if self.control_gvf is not None:
                if self.control_gvf.learner.finished_episode:
                    self.reset_episode()

            # save values
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


