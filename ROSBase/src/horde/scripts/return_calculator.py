#!/usr/bin/env python

"""
Author: Banafsheh Rafiee

Description:
ReturnCalculator samples some time steps from the behavior policy and computes the return for them. 
In order to compute the return for each sample time step, it switches from the behavior policy to the target policy. 
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
from state_representation import StateConstants
from tools import timing
from visualize_pixels import Visualize
from std_msgs.msg import Bool

class ReturnCalculator:

    def __init__(self, 
                 time_scale,
                 gvf,
                 num_features, 
                 features_to_use, 
                 behavior_policy,
                 target_policy):
        
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
        self.topics = topics

        rospy.loginfo("Started sensor threads.")

        # smooth out the actions
        self.time_scale = time_scale
        self.r = rospy.Rate(1.0/self.time_scale)        

        # agent info
        self.gvf             = gvf
        self.target_policy   = target_policy
        self.behavior_policy = behavior_policy

        self.state_manager = StateManager(features_to_use)
        self.feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        self.img_to_cv2 = CvBridge().compressed_imgmsg_to_cv2

        # information for managing the shift between the target and behavior policies
        self.following_mu = 0
        self.following_pi = 1
        self.current_condition = self.following_mu
        
        self.current_policy = self.behavior_policy
        self.fixed_steps_under_pi = 100
        self.fixed_step_under_mu = 100
        self.mu_max_horizon      = 100
        self.steps_under_mu = self.fixed_step_under_mu + np.random.randint(self.mu_max_horizon)


        # MSRE information
        self.sample_size = 1000
        self.samples_phi = np.zeros((self.sample_size, num_features))
        self.samples_G   = np.zeros(self.sample_size)

        self.cumulant_buffer  = np.zeros(self.fixed_steps_under_pi)
        self.gamma_buffer     = np.zeros(self.fixed_steps_under_pi)

        # Set up publishers
        action_publisher = rospy.Publisher('action_cmd', 
                                           geom_msg.Twist,
                                           queue_size=1)
        termination_publisher = rospy.Publisher('termination', 
                                                Bool,
                                                queue_size=1)
        self.publishers = {'action': action_publisher, 'termination': termination_publisher}
        rospy.loginfo("Done LearningForeground init.")

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
        # data['weights'] = self.gvfs[0].learner.theta if self.gvfs else None
        phi = self.state_manager.get_phi(**data)

        if 'last_action' in self.features_to_use:
            last_action = np.zeros(self.behavior_policy.action_space.size)
            last_action[self.behavior_policy.last_index] = 1
            phi = np.concatenate([phi, last_action])


        observation = self.state_manager.get_observations(**data)
        # observation['action'] = self.last_action

        return phi, observation


    def take_action(self, action):
        self.publishers['action'].publish(action)

    def update_return_buffers(self, index, observations):
        self.cumulant_buffer[index] = self.gvf.cumulant(observations)
        self.gamma_buffer[index] = self.gvf.gamma(observations)

    def compute_return(self, sample_number):
        G = 0.0
        for i in range(self.fixed_steps_under_pi):
            gamma = 1
            for j in range(i):
                gamma = gamma * self.gamma_buffer[j]
            G = G + gamma * self.cumulant_buffer[i]
        print "----------------------------------"
        print "computed return: ", G  
        print "----------------------------------"
          
        self.samples_G[sample_number] = G

    def run(self):

        sample_number = 0
        num_steps_followed_mu = 0
        num_steps_followed_pi = 0

        while not rospy.is_shutdown():

            start_time = time.time()
            # get new state
            phi, observations = self.create_state()

            print "-------------------------------------"
            print "num_steps_followed_mu:", num_steps_followed_mu
            print "num_steps_followed_pi:", num_steps_followed_pi
            print "current_condition:", self.current_condition
            print "fixed_steps_under_pi:", self.fixed_steps_under_pi
            print "steps_under_mu:", self.steps_under_mu
            print "-------------------------------------"

            # take action
            self.current_policy.update(phi, observations)
            action = self.current_policy.choose_action()
            self.take_action(action)

            # update cumulant and gamma buffers if following the target policy
            if self.current_condition == self.following_pi:
                self.update_return_buffers(index = num_steps_followed_pi, observations = observations)
                num_steps_followed_pi += 1
            elif self.current_condition == self.following_mu:
                num_steps_followed_mu += 1

            # figure out which policy should be followed
            if num_steps_followed_pi == self.fixed_steps_under_pi:
                self.current_condition = self.following_mu
                self.current_policy = self.behavior_policy
                self.steps_under_mu = self.fixed_step_under_mu + np.random.randint(self.mu_max_horizon)
                num_steps_followed_pi = 0

                # compute and store return
                self.compute_return(sample_number)
                sample_number = sample_number + 1
                print "sample_number:", sample_number

            elif num_steps_followed_mu == self.steps_under_mu:
                self.current_condition = self.following_pi
                self.current_policy = self.target_policy
                num_steps_followed_mu = 0
                
                # store phi
                self.samples_phi[sample_number, :] = phi[self.feature_indices]

            # terminate if collected information for sample size
            if (sample_number % 10) == 0 and num_steps_followed_pi == (self.fixed_steps_under_pi - 1):
                np.savez("actual_return_{}.npz".format(self.gvf.name), _return = self.samples_G, samples = self.samples_phi)
            if sample_number == self.sample_size:
                np.savez("actual_return_{}.npz".format(self.gvf.name), _return = self.samples_G, samples = self.samples_phi)
                self.publishers["termination"].publish(True)
                break

            # sleep until next time step
            self.r.sleep()




def start_return_calculator(time_scale,
                            GVF,
                            num_features,
                            features_to_use,
                            behavior_policy,
                            target_policy):
    try:
        return_calculator = ReturnCalculator(time_scale,
                                             GVF,
                                             num_features,
                                             features_to_use,
                                             behavior_policy,
                                             target_policy)
        return_calculator.run()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
    