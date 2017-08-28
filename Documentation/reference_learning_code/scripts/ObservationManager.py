#!/usr/bin/env python

"""
Observation Manager code. Things to note:
 - hardcoded values in init.
 - what is the function of jsonData.json?
 - might need to change topic names
"""

"""
Author: David Quail, February, 2017.

Description:
The observation manager will
1. listen for various different sensor data coming from various different data sources (servos, cameras, etc).
2. Maintain the data structure of most recent observations across data sources sources.
3. Publish the most recent observations at a predictable time interval.

By massaging the sensory data and publishing it at frequent, predictable time intervals, a learning system can subscribe
to this message. Once subscribed, the learner can perform control and learning at this interval, rather than having
to poll for the information after actions.

TODO:
Publish a message of specific format that encapsulates all the data of interest. Not just a primitive int of angle
"""


import rospy
import threading
import yaml

from std_msgs.msg import String
from std_msgs.msg import Int16
from horde.msg import StateRepresentation


from dynamixel_driver.dynamixel_const import *

from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue

from dynamixel_msgs.msg import MotorState
from dynamixel_msgs.msg import MotorStateList

from TileCoder import *

import json

"""
sets up the subscribers and starts to broadcast the results in a thread every 0.1 seconds
"""

class ObservationManager:

    def __init__(self):
        self.publishingFrequency = 1.1 #Seconds between updates

        #Initialize the sensory values of interest
        self.motoEncoder = 0
        self.maxEncoder = 1023.0
        self.minEncoder = 510.0

        self.speed = 0
        self.maxSpeed = 200.0
        self.minSpeed = -200.0

        self.load = 0
        self.timestamp = 0

        self.lastX = [0.0] * TileCoder.numberOfTiles * TileCoder.numberOfTilings * TileCoder.numberOfTilings

        self.file = open('jsonData.json', 'w')

    """
    motorStatesCallback(self, data)
    Dynamixel callback
    Data of format:

    motor_states:
  -
        timestamp: 1485931061.8
        id: 2
        goal: 805
        position: 805
        error: 0
        speed: 0
        load: 0.0
        voltage: 12.3
        temperature: 32
        moving: False
      -
        timestamp: 1485931061.8
        id: 3
        goal: 603
        position: 603
        error: 0
        speed: 0
        load: 0.0
        voltage: 12.3
        temperature: 34
        moving: False

    """
    def motorStatesCallback(self, data):
        print("In observation_manager callback")

        self.motoEncoder = data.motor_states[0].position
        self.speed = data.motor_states[0].speed
        self.load = data.motor_states[0].load
        self.timestamp = data.motor_states[0].timestamp

        # print this to a file
        jsonData = {"speed": data.motor_states[0].speed, "position": data.motor_states[0].position, "load":data.motor_states[0].load, "voltage":data.motor_states[0].voltage, "temperature": data.motor_states[0].temperature, "timestamp": data.motor_states[0].timestamp}
        json.dump(jsonData, self.file)
        self.file.write('\n')

    def publishObservation(self):
        print("In publish observation")
        pubObservation = rospy.Publisher('observation_manager/state_update', StateRepresentation, queue_size = 10)
        msg = StateRepresentation()
        msg.speed = self.speed
        msg.encoder = self.motoEncoder
        msg.load = self.load
        msg.timestamp = self.timestamp

        #Create the feature vector
        featureVector = TileCoder.getFeatureVectorFromValues([((self.motoEncoder- self.minEncoder)/(self.maxEncoder-self.minEncoder)) * TileCoder.numberOfTiles, ((self.speed + self.maxSpeed) / (self.maxSpeed - self.minSpeed)) * TileCoder.numberOfTiles])
        msg.lastX = self.lastX
        msg.X = featureVector

        self.lastX = featureVector

        pubObservation.publish(msg)

        threading.Timer(self.publishingFrequency, self.publishObservation).start()

    def start(self):
        rospy.init_node('observation_manager', anonymous=True)
        # Subscribe to all of the relevent sensor information. To start, we're only interested in motor_states, produced by the dynamixels
        rospy.Subscriber("motor_states/pan_tilt_port", MotorStateList, self.motorStatesCallback)

        self.publishObservation()

if __name__ == '__main__':
    manager = ObservationManager()
    manager.start()


