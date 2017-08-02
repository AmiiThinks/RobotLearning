#!/usr/bin/env python

"""
Author: Banafsheh Rafiee

Description:
ReturnCalculator samples some time steps from the behavior policy and computes the return for them. 
In order to compute the return for each sample time step, it switches from the behavior policy to the target policy. 
"""

from cv_bridge.core import CvBridge
import numpy as np
import geometry_msgs.msg as geom_msg
from Queue import Queue
import rospy
import std_msgs.msg as std_msg
import threading
import sys
import time
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool

from gvf import GVF
from state_representation import StateManager
from state_representation import StateConstants
from gentest_state_representation import GenTestStateManager
import tools
from tools import timing
from visualize_pixels import Visualize


class ReturnCalculator:

    def __init__(self, 
                 time_scale,
                 gvf,
                 num_features, 
                 features_to_use, 
                 behavior_policy,
                 target_policy):
        
        self.features_to_use = features_to_use + ['core']
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
        self.sample_size = 2
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
        

        phi = self.state_manager.get_phi(**data)

        # update the visualization of the image data
        # self.visualization.update_colours(image_data)

        # takes a long time, only uncomment if necessary
        # rospy.loginfo(phi)

        observation = self.state_manager.get_observations(**data)
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

        # Keep track of time for when to avoid sleeping
        sleep_time = self.time_scale - 0.0001
        tic = time.time()

        while not rospy.is_shutdown():

            while time.time() < tic:
                time.sleep(0.0001)

            # get new state
            phi, observations = self.create_state()

            print "-------------------------------------"
            print "num_steps_followed_mu:", num_steps_followed_mu
            print "num_steps_followed_pi:", num_steps_followed_pi
            print "current_condition:", self.current_condition
            print "fixed_steps_under_pi:", self.fixed_steps_under_pi
            print "steps_under_mu:", self.steps_under_mu
            if bool(sum(observations["bump"])):
                print "bumped"
            print "-------------------------------------"

            # take action
            self.current_policy.update(phi, observations)
            action = self.current_policy.choose_action()
            self.take_action(action)

            # reset tic
            tic += sleep_time

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
            if sample_number == self.sample_size:
                np.savez("actual_return_{}.npz".format(self.gvf.name), _return = self.samples_G, samples = self.samples_phi)
                self.publishers["termination"].publish(True)
                break



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
    