#!/usr/bin/env python

"""
Author: Michele Albach, David Quail, Parash Rahman, Niko Yasui, June 1, 2017.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""
from __future__ import division

import geometry_msgs.msg as geom_msg
import numpy as np
import os, sys
import pickle
import rospy
import random
import std_msgs.msg as std_msg
import subprocess
import sys
import threading
import time
import tools

from cv_bridge.core import CvBridge
from gentest_state_representation import GenTestStateManager
from geometry_msgs.msg import Twist, Vector3
from gvf import GVF
from Queue import Queue
from state_representation import StateManager
from tools import timing
from visualize_pixels import Visualize

class LearningForeground:
    def __init__(self,
                 time_scale,
                 gvfs,
                 features_to_use,
                 behavior_policy,
                 control_gvf=None):
        
        # Initiates this session's cumulant tallies with 'T:' 
        self.observation_file = open('observations.txt', 'ab+')
        self.observation_file.write('T:')
        self.observation_file.close()

        self.vis = False

        self.features_to_use = set(features_to_use + ['core', 'ir'])
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

        rospy.loginfo("Started sensor threads.")

        # smooth out the actions
        self.time_scale = time_scale
        self.r = rospy.Rate(1.0/self.time_scale)

        # agent info
        self.gvfs = gvfs
        self.control_gvf = control_gvf
        self.behavior_policy = behavior_policy
        self.avg_td_err = None

        self.state_manager = StateManager(features_to_use)
        self.img_to_cv2 = CvBridge().compressed_imgmsg_to_cv2

        # currently costs about 0.0275s per timestep
        rospy.loginfo("Creating visualization.")

        if self.vis:
            self.visualization = Visualize(self.state_manager.pixel_mask,
                                           imsizex=640,
                                           imsizey=480)

        rospy.loginfo("Done creatiing visualization.")

        # previous timestep information
        self.last_action = None
        self.last_phi = None
        self.preds = {g:None for g in self.gvfs}
        self.last_observation = None
        self.last_mu = 1

        # Set up publishers
        pub_name = lambda g, lab: '{}/{}'.format(g, lab) if g else lab
        pub = lambda g, lab: rospy.Publisher(pub_name(g, lab), 
                                             std_msg.Float64, 
                                             queue_size=10)
        action_publisher = rospy.Publisher('action_cmd2', 
                                           geom_msg.Twist,
                                           queue_size=1)
        pause_publisher = rospy.Publisher('pause', 
                                          std_msg.Bool,
                                          queue_size=1)
        termination_publisher = rospy.Publisher('termination', 
                                                std_msg.Bool,
                                                queue_size=1)

        self.publishers = {'action': action_publisher,
                           'pause': pause_publisher,
                           'termination': termination_publisher
                          }
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
            self.publishers[gvf]['prediction'].publish(self.preds[gvf])
            self.publishers[gvf]['cumulant'].publish(gvf.last_cumulant)
            self.publishers[gvf]['td_error'].publish(gvf.evaluator.td_error)
            self.publishers[gvf]['avg_td_error'].publish(gvf.evaluator.avg_td_error)
            self.publishers[gvf]['rupee'].publish(gvf.evaluator.rupee)

    @timing
    def create_state(self):
        # bumper constants from http://docs.ros.org/hydro/api/kobuki_msgs/html/msg/SensorState.html
        bump_codes = [1, 4, 2]

        # initialize data
        additional_features = set(tools.features.keys() + ['charging'])
        sensors = self.features_to_use.union(additional_features)

        # build data (used to make phi)
        data = {sensor: None for sensor in sensors}
        for source in sensors - set(['ir']):
            temp = None
            try:
                while True:
                    temp = self.recent[tools.features[source]].get_nowait()
            except:
                pass
            data[source] = temp

        temp = []
        try:
            while True:
                temp.append(self.recent[tools.features['ir']].get_nowait())
        except:
            pass

        # use only the last 10 values, helpful at the end of episode when we have accumulated at lot or IR data
        data['ir'] = temp[-10:] if temp else None

        if data['core'] is not None:
            bump = data['core'].bumper
            data['bump'] = map(lambda x: bool(x & bump), bump_codes)
            data['charging'] = bool(data['core'].charger & 2)
        if data['ir'] is not None:
            ir = [[0]*6]*3
            # bitwise 'or' of all the ir data in last time_step
            for temp in data['ir']:
                a = [[int(x) for x in format(temp, '#08b')[2:]] for temp in [ord(obs) for obs in temp.data]]
                ir = [[k | l for k, l in zip(i, j)] for i, j in zip(a, ir)]
            data['ir'] = [int(''.join([str(i) for i in ir_temp]),2) for ir_temp in ir] 
        if data['image'] is not None:
            data['image'] = np.asarray(self.img_to_cv2(data['image']))
        if data['odom'] is not None:
            pos = data['odom'].pose.pose.position
            data['odom'] = np.array([pos.x, pos.y])
        if data['imu'] is not None:
            data['imu'] = data['imu'].orientation.z
        if 'bias' in self.features_to_use:
            data['bias'] = True
        data['weights'] = self.gvfs[0].learner.theta if self.gvfs else None
        phi = self.state_manager.get_phi(**data)

        if 'last_action' in self.features_to_use:
            last_action = np.zeros(self.behavior_policy.action_space.size)
            last_action[self.behavior_policy.last_index] = 1
            phi = np.concatenate([phi, last_action])

        # update the visualization of the image data
        if self.vis:
            self.visualization.update_colours(data['image'])

        observation = self.state_manager.get_observations(**data)
        observation['action'] = self.last_action

        return phi, observation

    def take_action(self, action):
        self.publishers['action'].publish(action)

    def reset_episode(self):
        # temp = random.randint(0,50)
        # for i in range(temp):
        #     if i < temp:
        #         self.take_action(Twist(Vector3(-0.1, 0, 0), Vector3(0, 0, 0)))
        #     # elif i >= 50 and i < 50+temp:
        #     #     self.take_action(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.5)))
        #     # else:
        #     #     self.take_action(Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0)))
        #     rospy.loginfo('taking random action number: {}'.format(i))
        #     self.r.sleep()
        # self.publishers["pause"].publish(True)
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # interrupt = os.path.join(dir_path, 'interrupt_auto_docking.py')
        # os.system('python {}'.format(interrupt))
        # self.publishers["pause"].publish(False)

        for i in range(random.randint(0,60)):
            action, mu = self.gvfs[0].learner.take_random_action()
            self.take_action(action)
            rospy.loginfo('taking random action number: {}'.format(i))
            self.r.sleep()        
    
    def run(self):
        while not rospy.is_shutdown():
            start_time = time.time()

            # get new state
            phi_prime, observation = self.create_state()

            if (observation['bump']):
                # adds a tally for the added cumulant
                self.observation_file = open('observations.txt', 'ab+')
                self.observation_file.write('1')
                self.observation_file.close()

            # select and take an action
            self.behavior_policy.update(phi_prime, observation)
            action = self.behavior_policy.choose_action()
            mu = self.behavior_policy.get_probability(action)
            self.take_action(action)

            # make prediction
            self.preds = {g:g.predict(phi_prime, action) for g in self.gvfs}

            # learn
            if self.last_observation is not None:
                self.update_gvfs(phi_prime, observation)

            # check if episode is over
            if self.control_gvf is not None:
                if self.control_gvf.learner.finished_episode(self.control_gvf.last_cumulant):
                    self.reset_episode()

            # save values
            self.last_phi = phi_prime if len(phi_prime) else None
            self.last_action = action
            self.last_mu = self.behavior_policy.get_probability(action)
            self.last_observation = observation

            # timestep logging
            total_time = time.time() - start_time
            time_msg = "Current timestep took {:.4f} sec.".format(total_time)
            rospy.loginfo(time_msg)
            if total_time > self.time_scale:
                if self.control_gvf is not None:
                    if not self.control_gvf.learner.finished_episode(self.control_gvf.last_cumulant):
                        rospy.logerr("Timestep took too long!")
                else:
                    rospy.logerr("Timestep took too long!")

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


