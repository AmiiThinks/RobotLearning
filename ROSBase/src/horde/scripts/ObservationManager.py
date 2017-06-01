#!/usr/bin/env python

"""
Author: David Quail, Niko Yasui, June 1, 2017.

Description:
The observation manager will:
1. Compile the most recent sensor observations across sensors using
   the SensorParser class.
2. Maintain the data structure of most recent observations across data
   sources sources.
3. Publish the most recent observations at a predictable time interval.

By massaging the sensory data and publishing it at frequent, predictable 
time intervals, a learning system can subscribe to this message. Once
subscribed, the learner can perform control and learning at this 
interval, rather than having to poll for the information after actions.
"""

from cv_bridge.core import CvBridge
import kobuki_msgs.msg as kob_msg
import math
import msgpack
import numpy as np
from Queue import Queue
import rospy
import signal
import sensor_msgs.msg as sens_msg
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg as std_msg
import sys
import threading
import time

from SensorParser import SensorParser
from Tools import merge_dicts

# for some reason we need this to be able to ctrl-c the program, but
# there is probably a more graceful way
signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))

"""
sets up the subscribers and starts to broadcast the results in a thread every 0.1 seconds
"""

thread = False

class ObservationManager:

    def __init__(self, topics, publishing_frequency):

        # learning rate
        self.publishing_frequency = publishing_frequency

        # list of topics
        self.topics = topics 

        # create a dictionary (threadsafe) for storing recent data
        self.most_recent = {topic:dict() for topic in topics}

    def publish_observations(self):

        # Keep track of time for when to avoid sleeping
        self.publishing_frequency -= 0.0001
        tic = time.time()

        while True:
            # To avoid the drift of just calling time.sleep()
            while time.time() < tic:
                time.sleep(0.0001)

            # create the message
            msg = std_msg.String()
            msg.data = msgpack.packb(merge_dicts(*self.most_recent.values()))

            # publish the message
            self.publisher.publish(msg)

            # reset tic
            tic += self.publishing_frequency

    def start_thread(self, topic):
        sp = SensorParser(topic)
        sp.start(self.most_recent[topic])

    def start(self):
        rospy.init_node('observation_manager', anonymous=True)
        self.publisher = rospy.Publisher('observation_manager/combined', 
                                         std_msg.String,
                                         queue_size=1)

        # setup sensor parsers
        for topic in self.topics:
            t = threading.Thread(target=self.start_thread, args=[topic])
            t.setDaemon(True)
            t.start()

        # publish the most recent observation
        self.publish_observations()

        rospy.spin()

if __name__ == '__main__':
    try:
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
        manager = ObservationManager(topics, 0.2)
        manager.start()
    except rospy.ROSInterruptException:
        pass

