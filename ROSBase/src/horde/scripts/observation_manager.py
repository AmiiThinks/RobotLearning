#!/usr/bin/env python

"""
Author: David Quail, Niko Yasui, June 1, 2017.

Description:
The observation manager will:
1. Compile the most recent sensor observations across sensors using
   the SensorParser class.
2. Maintain the data structure of most recent observations across data
   sources sources.
"""

import rospy
import std_msgs.msg as std_msg
import threading

from sensor_parser import SensorParser

class ObservationManager:

    def __init__(self, topics, dictionary):

        # list of topics
        self.topics = topics 

        self.start(dictionary)

    def start_thread(self, topic, dictionary):
        sp = SensorParser(topic)
        sp.start(dictionary)

    def start(self, dictionary):
        self.publisher = rospy.Publisher('observation_manager/combined', 
                                         std_msg.String,
                                         queue_size=1)

        # setup sensor parsers
        for topic in self.topics:
            t = threading.Thread(target=self.start_thread, 
                                 args=(topic, dictionary))
            t.daemon = True
            t.start()

        # keep process alive
        rospy.spin()

