#!/usr/bin/env python

"""
Author: David Quail, February, 2017.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""


import rospy
import threading
import yaml

from std_msgs.msg import String
from std_msgs.msg import Int16
from std_msgs.msg import Float64

from dynamixel_driver.dynamixel_const import *

from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue

from dynamixel_msgs.msg import MotorState
from dynamixel_msgs.msg import MotorStateList

from horde.msg import StateRepresentation

from BehaviorPolicy import *
from TileCoder import *
from GVF import *
from ActorCritic import *
from ActorCriticContinuous import *
from Verifier import *
from PredictLoadDemon import *
import time

import numpy

"""
sets up the subscribers and starts to broadcast the results in a thread every 0.1 seconds
"""
alpha = 0.1

def directLeftPolicy(state):
    return 2

def atLeftGamma(state):
    if state.encoder >=1020.0:
        return 0
    else:
        return 1

def loadCumulant(state):
    return state.load

def encoderCumulant(state):
    return state.encoder

def timestepCumulant(state):
    return 1

def makeGammaFunction(gamma):
    def gammaFunction(state):
        return gamma
    return gammaFunction

def makeVectorBitCumulantFunction(bitIndex):
    def cumulantFunction(state):
        if (state.X[bitIndex] == 1):
            return 1
        else:
            return 0
    return cumulantFunction


def createNextEncoderGVF():
    gvfs = []
    gvOffPolicy = GVF(TileCoder.numberOfTiles * TileCoder.numberOfTiles * TileCoder.numberOfTilings,
                      alpha / TileCoder.numberOfTilings, isOffPolicy=True, name="PredictNextEncoderOffPolicy")

    gvOffPolicy.cumulant = encoderCumulant
    gvOffPolicy.policy = directLeftPolicy
    gvfs.append(gvOffPolicy)
    return gvfs


def createPredictLoadGVFs():
    #GVFS that predict how much future load at different timesteps on and off policy
    #Create On policy GVFs for gamma values that correspond to timesteps: {1,2,3,4,5,6,7,8,9,10}.
    #Create Off policy GVFs for the same gamma values

    gvfs = []

    for i in range(1, 10, 1):
        #T = 1/(1-gamma)
        #gamma = (T-1)/T
        gamma = (i-1.0)/i

        #Create On policy gvf

        gvfOnPolicy = GVF(TileCoder.numberOfTiles*TileCoder.numberOfTiles * TileCoder.numberOfTilings, alpha / TileCoder.numberOfTilings, isOffPolicy = False, name = "PredictedLoadGammaOnPolicy" + str(i))
        gvfOnPolicy.gamma = makeGammaFunction(gamma)
        gvfOnPolicy.cumulant = loadCumulant

        gvfs.append(gvfOnPolicy)

        #Create Off policy gvf
        gvOffPolicy = GVF(TileCoder.numberOfTiles*TileCoder.numberOfTiles * TileCoder.numberOfTilings, alpha / TileCoder.numberOfTilings, isOffPolicy = True, name = "PredictedLoadGammaOffPolicy" + str(i))
        gvOffPolicy.gamma = makeGammaFunction(gamma)
        gvOffPolicy.cumulant = loadCumulant
        gvOffPolicy.policy = directLeftPolicy

        gvfs.append(gvOffPolicy)

    return gvfs

def createHowLongUntilLeftGVFs():
    #Create GVFs that predict how long it takes to get to the end. One on policy. And one off policy - going straight there.

    gvfs = []

    gvfOn = GVF(TileCoder.numberOfTiles*TileCoder.numberOfTiles * TileCoder.numberOfTilings, alpha / TileCoder.numberOfTilings, isOffPolicy = False, name = "HowLongLeftOnPolicy")
    gvfOn.gamma = atLeftGamma
    gvfOn.cumulant = timestepCumulant

    gvfs.append(gvfOn)

    gvfOff = GVF(TileCoder.numberOfTiles * TileCoder.numberOfTiles * TileCoder.numberOfTilings, alpha / TileCoder.numberOfTilings, isOffPolicy=True, name = "HowLongLeftOffPolicy")
    gvfOff.gamma = atLeftGamma
    gvfOff.cumulant = timestepCumulant
    gvfOff.policy = directLeftPolicy

    gvfs.append(gvfOff)

    return gvfs

def createNextBitGVFs():
    gvfs = []

    #TODO Remove after testing
    """
    gvfOn = GVF(TileCoder.numberOfTiles * TileCoder.numberOfTiles * TileCoder.numberOfTilings,
                0.1 / TileCoder.numberOfTilings, isOffPolicy=False, name="NextBitOnPolicy" + str(301))
    gvfOn.cumulant = makeVectorBitCumulantFunction(301)
    gvfs.append(gvfOn)
    """
    #TODO uncomment after testing


    vectorSize = TileCoder.numberOfTiles * TileCoder.numberOfTiles * TileCoder.numberOfTilings

    for i in range(0, vectorSize, 1):
        gvfOn = GVF(TileCoder.numberOfTiles*TileCoder.numberOfTiles * TileCoder.numberOfTilings, alpha / TileCoder.numberOfTilings, isOffPolicy = False, name = "NextBitOnPolicy"+ str(i))
        gvfOn.cumulant = makeVectorBitCumulantFunction(i)
        gvfs.append(gvfOn)

        gvfOff = GVF(TileCoder.numberOfTiles * TileCoder.numberOfTiles * TileCoder.numberOfTilings, alpha / TileCoder.numberOfTilings, isOffPolicy=True, name = "NextBitOffPolicy"+ str(i))
        gvfOff.cumulant = makeVectorBitCumulantFunction(i)
        gvfOff.policy = directLeftPolicy
        gvfs.append(gvfOff)


    return gvfs

def createActorCritic():
    ac = ActorCritic()
    return ac

def createActorCriticContinuous():
    ac = ActorCriticContinuous()
    return ac

class LearningForeground:

    def __init__(self):
        self.demons = []
        self.verifiers = {}

        self.behaviorPolicy = BehaviorPolicy()

        self.lastAction = 0

        self.currentRadians = 0
        self.increasingRadians = True

        #Initialize the demons appropriately depending on what test you are runnning by commenting / uncommenting
        self.pavlovDemon = False
        self.actorCritic = False
        #self.demons = createPredictLoadGVFs()
        #self.demons = createHowLongUntilLeftGVFs()
        #self.actorCritic = createActorCritic()
        #self.actorCritic = createActorCriticContinuous()
        #self.demons = createNextBitGVFs()

        #self.demons = createNextEncoderGVF()
        #self.pavlovDemon = self.demons[0]

        self.previousState = False

        #Initialize the sensory values of interest

    def performPavlov(self):
        print("!!!Pavlov control!!!!")
        self.lastAction = 1
        self.currentRadians = self.currentRadians - 0.5
        self.increasingRadians = False
        print("Switching direction")
        print("Going to radians: " + str(self.currentRadians))
        pub = rospy.Publisher('tilt_controller/command', Float64, queue_size=10)
        pub.publish(self.currentRadians)

    def performSlowBackAndForth(self):
        if self.increasingRadians:
            self.lastAction = 2
        else:
            self.lastAction = 1

        if (self.increasingRadians):
            self.currentRadians = self.currentRadians + 0.05
            if self.currentRadians >= 3.0:
                print("Switching direction!!!")
                self.increasingRadians = False
        else:
            self.currentRadians = self.currentRadians - 0.05
            if self.currentRadians <= 0.0:
                print("Switching direction!!!")
                self.increasingRadians = True

        print("Going to radians: " + str(self.currentRadians))
        pub = rospy.Publisher('tilt_controller/command', Float64, queue_size=10)
        pub.publish(self.currentRadians)

    def performContinuousAction(self, action):
        # bound this to an action value between 510 and 1023
        # action is bounded between -10 and 10
        self.lastAction = action

        action = action + 10
        pct = action / 20  # gets how close to the extreme it is
        radians = 3.0 * pct
        if radians < 0.0:
            radians = 0.0
        elif radians > 3.0:
            radians = 3.0
        pub = rospy.Publisher('tilt_controller/command', Float64, queue_size=10)
        pub.publish(radians)

    def performAction(self, action):
        print("Performing action: "  + str(action))
        #Take the action and issue the actual dynamixel command
        pub = rospy.Publisher('tilt_controller/command', Float64, queue_size=10)

        if (action ==1):
            #Move left
            pub.publish(0.0)
        elif (action == 2):
            pub.publish(3.0)
        elif (action >=510) & (action <= 1023):
            pct = (action - 510) / (1023 - 510)
            radians = 3.0 * pct
            pub.publish(radians)

        self.lastAction = action

    def updateActorCritic(self, newState):
        encoderPosition = newState.encoder
        speed = newState.speed
        load = newState.load

        if self.previousState:
            #Learning
            if self.actorCritic:
                self.actorCritic.learn(self.previousState, self.lastAction, newState)

    def updateDemons(self, newState):
        print("LearningForeground received stateRepresentation encoder: " + str(newState.encoder) + ", speed: " + str(newState.speed))

        encoderPosition = newState.encoder
        speed = newState.speed
        load = newState.load

        if self.previousState:
            #Learning
            for demon in self.demons:
                predBefore = demon.prediction(self.previousState)
                demon.learn(self.previousState, self.lastAction, newState)
                print("Demon prediction before: " + str(predBefore))
                print("Demon prediction after: " + str(demon.prediction(self.previousState)))
                if demon in self.verifiers:
                    self.verifiers[demon].append(demon.gamma(newState), demon.cumulant(newState), demon.prediction(newState), newState)


    def publishPredictionsAndErrors(self, state):
        averageRupee = 0
        averageUDE = 0
        i = 1
        for demon in self.demons:
            pred = demon.prediction(state)
            rupee = demon.rupee()
            averageRupee = averageRupee + (1.0 / i) * (rupee - averageRupee)

            ude = demon.ude()
            averageUDE = averageUDE + (1.0 / i) * (ude - averageUDE)

            i = i + 1

            pubPrediction = rospy.Publisher('horde_verifier/' + demon.name + 'Prediction', Float64, queue_size=10)
            pubPrediction.publish(pred)

            pubRupee = rospy.Publisher('horde_verifier/' + demon.name + 'Rupee', Float64, queue_size=10)
            pubRupee.publish(rupee)
            pubUDE = rospy.Publisher('horde_verifier/' + demon.name + 'UDE', Float64, queue_size=10)
            pubUDE.publish(ude)

        avgRupee = rospy.Publisher('horde_verifier/AverageRupee', Float64, queue_size=10)
        avgRupee.publish(averageRupee)

        avgUDE = rospy.Publisher('horde_verifier/AverageUDE', Float64, queue_size=10)
        avgUDE.publish(averageUDE)



    def receiveStateUpdateCallback(self, newState):
        #Staterepresentation callback

        #publish new state encoder
        #TODO Remove after testing magic 301
        #e = (newState.encoder - 510.0) / 5.0
        e = (newState.encoder)
        e = 100 * (e - 510.0) / (1023.0 - 510.0)
        pubCumulant = rospy.Publisher('horde_verifier/EncoderPosition', Float64, queue_size=10)
        pubCumulant.publish(e)
        #1. Learn
        #Convert the list of X's into an actual numpy array
        newState.X = numpy.array(newState.X)
        newState.lastX = numpy.array(newState.lastX)
        startTime = time.time()
        self.updateDemons(newState)
        self.updateActorCritic(newState)
        endTime = time.time()

        #0.09 seconds on average. Clobbering CPU
        #2. Take action
        #TODO - place code for pavlov
        pavlovSignal = False
        if self.pavlovDemon:
            pred = self.pavlovDemon.prediction(newState)
            print("pavlov prediction:" + str(pred))
            pavlovSignal = self.pavlovDemon.prediction(newState) > 900.0

        if pavlovSignal == True:
            self.performPavlov()
        else:
            #self.performSlowBackAndForth()
            #action = self.behaviorPolicy()
            if self.actorCritic:
                action = self.actorCritic.pickActionForState(newState)
                self.performContinuousAction(action)
            else:
                action  = self.behaviorPolicy.policy(newState)
                self.performAction(action)


        #3. Publish predictions and errors
        if self.previousState:
            self.publishPredictionsAndErrors(self.previousState)

        self.previousState = newState

    def start(self):
        print("In Horde foreground start")
        # Subscribe to all of the relevent sensor information. To start, we're only interested in motor_states, produced by the dynamixels
        #rospy.Subscriber("observation_manager/servo_position", Int16, self.receiveObservationCallback)
        rospy.Subscriber("observation_manager/state_update", StateRepresentation, self.receiveStateUpdateCallback)

        rospy.spin()

if __name__ == '__main__':
    foreground = LearningForeground()
    #Set the mixels to 0
    rospy.init_node('horde_foreground', anonymous=True)
    pub = rospy.Publisher('tilt_controller/command', Float64, queue_size=10)
    pub.publish(0.0)

    time.sleep(3)

    foreground.start()


"""
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