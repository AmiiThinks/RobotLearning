#!/usr/bin/env python

"""
Author: Michele Albach, David Quail, Parash Rahman, Niko Yasui, Shibhansh
Dohare, June 1, 2017.

Description:
LearningForeground contains a collection of GVF's. It accepts new state
representations, learns, and then takes action.

"""
from __future__ import division

import random
import time
from Queue import Queue
from multiprocessing import Value

import cv2
import geometry_msgs.msg as geom_msg
import numpy as np
import rosbag
import rospy
import std_msgs.msg as std_msg

from state_representation import StateManager
import tools
from tools import timing
from visualize_pixels import Visualize

class LearningForeground:
    def __init__(self,
                 time_scale,
                 gvfs,
                 features_to_use,
                 behavior_policy,
                 stats,
                 control_gvf=None,
                 cumulant_counter=None,
                 reset_episode=None):

        # function that generates a list of actions to perform to reset episode
        self.reset_episode = reset_episode 

        # set up ros
        rospy.init_node('agent', anonymous=True)

        self.COLLECT_DATA_FLAG = False

        # counts the total cumulant for the session
        if cumulant_counter:
            self.cumulant_counter = cumulant_counter
        else:
            self.cumulant_counter = Value('d', 0)

        # capture this session's data and actions
        if self.COLLECT_DATA_FLAG:
            self.history = rosbag.Bag('results.bag', 'w')
            self.current_time = rospy.Time().now()

        self.vis = False
        # self.vis = True

        extras = {'core', 'ir', 'odom'}
        self.features_to_use = set(features_to_use).union(extras)

        topics = filter(lambda x: x,
                        [tools.features[f] for f in self.features_to_use])

        # set up dictionary to receive sensor info
        self.recent = {topic: Queue(0) for topic in topics}

        # setup sensor parsers
        for topic in topics:
            rospy.Subscriber(topic,
                             tools.topic_format[topic],
                             self.recent[topic].put)

        rospy.loginfo("Started sensor threads.")

        # smooth out the actions
        self.time_scale = time_scale
        self.r = rospy.Rate(int(1.0 / self.time_scale))

        # agent info
        self.gvfs = gvfs
        self.control_gvf = control_gvf
        self.behavior_policy = behavior_policy
        self.avg_td_err = None

        self.state_manager = StateManager(features_to_use)

        if self.vis:
            rospy.loginfo("Creating visualization.")
            self.visualization = Visualize(self.state_manager.pixel_mask,
                                           imsizex=640,
                                           imsizey=480)
            rospy.loginfo("Done creatiing visualization.")

        # previous timestep information
        self.last_action = None
        self.last_phi = None
        self.last_observation = None
        self.last_mu = 1

        # experience replay
        self.to_replay_experience = False

        action_publisher = rospy.Publisher('action_cmd',
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

        valid_stats = ['prediction', 'td_error', 'avg_td_error', 'rupee',
                       'MSRE', 'cumulant', 'phi', 'e', 'rho', 'ESS']

        self.stat_data = {'prediction': lambda g: g.last_prediction,
                          'cumulant': lambda g: g.last_cumulant,
                          'td_error': lambda g: g.evaluator.td_error,
                          'avg_td_error': lambda g: g.evaluator.avg_td_error,
                          'rupee': lambda g: g.evaluator.rupee,
                          'MSRE': lambda g: g.evaluator.MSRE,
                          'phi': lambda g: g.phi.sum(),
                          'e': lambda g: g.learner.e.sum(),
                          'rho': lambda g: g.rho,
                          'ESS': lambda g: g.evaluator.ESS}
        self.stats = filter(lambda s: s in valid_stats, stats)

        def publisher_name(gvf, label):
            return '{}/{}'.format(gvf, label) if gvf else label

        def make_publisher(gvf, label):
            return rospy.Publisher(publisher_name(gvf, label),
                                   std_msg.Float64,
                                   queue_size=10)
        stat_publishers = {gvf: {stat: make_publisher(gvf.name, stat)
                                 for stat in self.stats}
                           for gvf in self.gvfs}
        self.publishers.update(stat_publishers)

        rospy.loginfo("Done LearningForeground init.")

    @timing
    def update_gvfs(self, phi_prime, observation, action):
        for gvf in self.gvfs:
            gvf.update(self.last_observation,
                       self.last_phi,
                       self.last_action,
                       observation,
                       phi_prime,
                       self.last_mu,
                       action)

        # publishing
        for gvf in self.gvfs:
            for stat in self.stats:
                self.publishers[gvf][stat].publish(self.stat_data[stat](gvf))

    def read_source(self, source, history=False):
        """Reads from the topics and returns the most recent value.
        """
        temp = [] if history else None
        try:
            stream = tools.features[source]
            while True:
                if history:
                    temp.append(self.recent[stream].get_nowait())
                else:
                    temp = self.recent[stream].get_nowait()
        except:
            pass
        return temp

    @timing
    def create_state(self):
        # bumper constants from
        # http://docs.ros.org/hydro/api/kobuki_msgs/html/msg/SensorState.html
        bump_codes = [1, 4, 2]

        # initialize data
        additional_features = set(tools.features.keys() + ['charging'])
        sensors = self.features_to_use.union(additional_features)

        # build data (used to make phi)
        data = {sensor: None for sensor in sensors}
        for source in sensors - {'ir', 'core'}:
            data[source] = self.read_source(source)

        data['ir'] = self.read_source('ir', history=True)[-10:]
        data['core'] = self.read_source('core', history=True)

        if data['core']:
            bumps = [dat.bumper for dat in data['core']]
            data['bump'] = np.sum(
                    [[bool(x & bump) for x in bump_codes] for bump in bumps],
                    axis=0, dtype=bool).tolist()
            data['charging'] = bool(data['core'][-1].charger & 2)

            # enter the data into rosbag
            if self.COLLECT_DATA_FLAG:
                for bindex in range(len(data['bump'])):
                    bump_bool = std_msg.Bool()
                    bump_bool.data = data['bump'][bindex] if data['bump'][
                        bindex] else False
                    self.history.write('bump' + str(bindex), bump_bool,
                                       t=self.current_time)
                charge_bool = std_msg.Bool()
                charge_bool.data = data['charging']
                self.history.write('charging', charge_bool,
                                   t=self.current_time)

        if data['ir']:
            ir = [[0] * 6] * 3
            # bitwise 'or' of all the ir data in last time_step
            for temp in data['ir']:
                a = [[int(x) for x in format(temp, '#08b')[2:]] for temp in
                     [ord(obs) for obs in temp.data]]
                ir = [[k | l for k, l in zip(i, j)] for i, j in zip(a, ir)]

            data['ir'] = [int(''.join([str(i) for i in ir_temp]), 2) for
                          ir_temp in ir]

            # enter the data into rosbag
            if self.COLLECT_DATA_FLAG:
                ir_array = std_msg.Int32MultiArray()
                ir_array.data = data['ir']
                self.history.write('ir', ir_array, t=self.current_time)

        if data['image'] is not None:
            # enter the data into rosbag
            # image_array = std_msg.Int32MultiArray()
            # image_array.data = data['image']
            if self.COLLECT_DATA_FLAG:
                self.history.write('image', data['image'], t=self.current_time)

            # uncompressed image
            data['image'] = np.fromstring(data['image'].data,
                                          np.uint8).reshape(480, 640, 3)

            # compressing image

        if data['cimage'] is not None:
            data['image'] = cv2.imdecode(np.fromstring(data['cimage'].data,
                                                       np.uint8),
                                         1)

        if data['odom'] is not None:
            pos = data['odom'].pose.pose.position
            lin_vel = data['odom'].twist.twist.linear.x
            ang_vel = data['odom'].twist.twist.angular.z
            data['odom'] = np.array([pos.x, pos.y, lin_vel, ang_vel])

            # enter the data into rosbag
            if self.COLLECT_DATA_FLAG:
                odom_x = std_msg.Float64()
                odom_x.data = pos.x
                odom_y = std_msg.Float64()
                odom_y.data = pos.y
                odom_lin = std_msg.Float64()
                odom_lin.data = lin_vel
                odom_ang = std_msg.Float64()
                odom_ang.data = ang_vel

                self.history.write('odom_x', odom_x, t=self.current_time)
                self.history.write('odom_y', odom_y, t=self.current_time)
                self.history.write('odom_lin', odom_lin, t=self.current_time)
                self.history.write('odom_ang', odom_ang, t=self.current_time)

        if data['imu'] is not None:
            data['imu'] = data['imu'].orientation.z

            # TODO: enter the  data into rosbag
        if 'bias' in self.features_to_use:
            data['bias'] = True
        data['weights'] = self.gvfs[0].learner.theta if self.gvfs else None
        phi = self.state_manager.get_phi(**data)

        if 'last_action' in self.features_to_use:
            last_action = np.zeros(self.behavior_policy.action_space.size)
            last_action[self.behavior_policy.last_index] = True
            phi = np.concatenate([phi, last_action])

            # update the visualization of the image data
        if self.vis:
            self.visualization.update_colours(data['image'])

        observation = self.state_manager.get_observations(**data)
        observation['action'] = self.last_action

        if observation['bump']:
            # adds a tally for the added cumulant
            self.cumulant_counter.value += 1

        return phi, observation

    def take_action(self, action):
        self.publishers['action'].publish(action)

    def run(self):
        avg_time = 0
        time_step = 0
        max_time = 0
        while not rospy.is_shutdown():
            start_time = time.time()
            self.current_time = rospy.Time().now()

            # get new state
            phi_prime, observation = self.create_state()

            # select and take an action
            self.behavior_policy.update(phi_prime, observation)
            action = self.behavior_policy.choose_action()
            mu = self.behavior_policy.get_probability(action)
            self.take_action(action)

            if self.COLLECT_DATA_FLAG:
                self.history.write('action', action, t=self.current_time)

            # learn
            if self.last_observation is not None:
                self.update_gvfs(phi_prime, observation, action)

            # check if episode is over and reset accordingly [episodic]
            if self.control_gvf is not None:
                if self.control_gvf.learner.episode_finished_last_step:
                    reset_actions = self.reset_episode()
                    for action in reset_actions:
                        self.take_action(action)
                        rospy.loginfo('taking random action number: {}'.format(action))
                        if self.to_replay_experience:
                            self.control_gvf.learner.uniform_experience_replay()
                        self.r.sleep()
                elif self.to_replay_experience:
                    # not to replay when the episode resets at it will also
                    # include the experience at the start of new episode
                    self.control_gvf.learner.uniform_experience_replay()

            # save values
            self.last_phi = phi_prime if len(phi_prime) else None
            self.last_action = action
            self.last_mu = mu
            self.last_observation = observation

            # timestep logging
            total_time = time.time() - start_time
            max_time = max(max_time, total_time)
            time_step += 1
            avg_time += (total_time - avg_time) / time_step
            time_msg = "Current timestep took {:.4f} sec.".format(total_time)
            rospy.loginfo(time_msg)

            if total_time > self.time_scale:
                if self.control_gvf is not None:
                    if not self.control_gvf.learner.episode_finished_last_step:
                        rospy.logerr("Timestep took too long!")
                else:
                    rospy.logerr("Timestep took too long!")

            # sleep until next time step
            self.r.sleep()
        if self.COLLECT_DATA_FLAG:
            self.history.close()


def start_learning_foreground(time_scale,
                              GVFs,
                              topics,
                              policy,
                              stats,
                              control_gvf=None,
                              cumulant_counter=None,
                              reset_episode=None):
    try:
        foreground = LearningForeground(time_scale,
                                        GVFs,
                                        topics,
                                        policy,
                                        stats,
                                        control_gvf,
                                        cumulant_counter,
                                        reset_episode)

        foreground.run()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
