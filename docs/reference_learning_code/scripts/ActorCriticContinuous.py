"""
Note: We probably won't have to use this.

Code for an Actor/Critic. Things to be aware of:
 - mean is bounded
 - variance is set to 5 if inf
 - hard-coded hyperparameters (add as args in init?)
 - self.minAction and self.maxAction (ad as args in init?)
 - reward function is in the class rather than a scalar being passed to the learner
"""

import numpy
import rospy
from std_msgs.msg import Float64
from horde.msg import StateRepresentation
from TileCoder import *

class ActorCriticContinuous():
    def __init__(self):
        self.maxAction = 1023.0
        self.minAction = 510.0
        self.numberOfFeatures = TileCoder.numberOfTilings * TileCoder.numberOfTiles * TileCoder.numberOfTiles

        #Eligibility traces and weights
        #Value / Critic
        self.elibibilityTraceValue = numpy.zeros(self.numberOfFeatures)
        self.valueWeights = numpy.zeros(self.numberOfFeatures)

        #Mean
        self.elibibilityTraceMean = numpy.zeros(self.numberOfFeatures)
        self.policyWeightsMean = numpy.zeros(self.numberOfFeatures)

        #Deviation
        self.elibibilityTraceVariance = numpy.zeros(self.numberOfFeatures)
        self.policyWeightsVariance = numpy.zeros(self.numberOfFeatures)

        self.lambdaPolicy = 0.35
        self.lambdaValue = 0.35
        self.averageReward = 0.0

        self.isRewardGoingLeft = True
        """
        self.stepSizeValue = 0.1
        self.stepSizeVariance = 0.01
        self.stepSizeMean = 10*self.stepSizeVariance
        self.rewardStep = 0.10
        """

        """
        #Patricks values
        self.stepSizeValue = 0.05
        self.stepSizeVariance = 0.005
        self.stepSizeMean = 0.005
        self.rewardStep = 0.0005
        """
        """
        self.stepSizeValue = 0.1
        self.stepSizeVariance = 0.01
        self.stepSizeMean = 0.02
        self.rewardStep = 0.05
        """
        self.stepSizeValue = 0.01
        self.stepSizeVariance = 0.01
        self.stepSizeMean = 0.1
        self.rewardStep = 0.005

        self.i = 0

    def mean(self, state):
        """
        Mean bounded in [-10, 10]
        """
        m = numpy.inner(self.policyWeightsMean, state.X)
        if m > 10.0:
            m = 10
        if m < -10.0:
            m = -10.0
        return m

    def variance(self, state):
        """
        Variance bounded away from zero and set to 5 if infinity. 
        """
        v = numpy.exp(numpy.inner(self.policyWeightsVariance, state.X))
        #on occasion, variance can be massive. so we bound it here
        if numpy.isinf(v):
            v = 5.0
        elif v < 0.001:
            v = 0.001
        return v

    def pickActionForState(self, state):

        print("******* pickActionForState ************")
        m = self.mean(state)
        v = self.variance(state) + 0.0000001 #to prevent against 0 variance
        pubMean = rospy.Publisher('horde_AC/Continuous/Mean', Float64, queue_size=10)
        pubMean.publish(m)

        pubVariance = rospy.Publisher('horde_AC/Continuous/Variance', Float64, queue_size=10)
        pubVariance.publish(v)

        print("mean: " + str(m) + ", variance: " + str(v))
        action = numpy.random.normal(m, v)
        #generally between -10 and 10. If greater or less than these values can cause overflow and underflow
        if ((state.encoder > 620) & (state.encoder < 700) & (state.speed <=0)):
            pubMeanSpecial = rospy.Publisher('horde_AC/Continuous/MeanInState', Float64, queue_size=10)
            pubMeanSpecial.publish(m)
            pubVarianceSpecial = rospy.Publisher('horde_AC/Continuous/VarianceInState', Float64, queue_size=10)
            pubVarianceSpecial.publish(v)

        if action < -10.0:
            action = -10.0 + 0.1*random.randint(1,4)

        if action > 10.0:
            action = 10.0 - 0.1*random.randint(1,4)

        print("action: " + str(action))

        pubAction = rospy.Publisher('horde_AC/Continuous/Action', Float64, queue_size=10)
        pubAction.publish(action)

        return action

    #This should probably not be defined with the actor critic, but rather be sent to the actor critic in the learn step
    def rewardOld(self, previousState, action, newState):
        if newState.encoder < 795:
            return 1
        else:
            return 0

    def reward(self, previousState, action, newState):

        #Higher reward, the closer you are to 550

        rewardGoingLeft = (1023.0 - newState.encoder) / 100.0
        rewardGoingRight = (newState.encoder - 510.0)/10.0
        print("---- reward iteration: " + str(self.i))
        self.i = self.i + 1

        #if self.i % 5000 == 0:
        if False:
            self.i = 1
            #flip every 150 steps (5 minutes)
            print("*************** flipping reward function **********")
            print("***************************************************")
            self.isRewardGoingLeft = not self.isRewardGoingLeft

        if self.isRewardGoingLeft:
            return rewardGoingLeft
        else:
            return rewardGoingRight


    def learn(self, previousState, action, newState):
        print("============= In Continuous actor critic learn =========")

        reward = self.reward(previousState, action, newState)

        print("previous encoder: " + str(previousState.encoder) + ", speed: " + str(previousState.speed) + ", new encoder: " + str(newState.encoder) + " speed: " + str(newState.speed) +  ", action: " + str(action) + ", reward: " + str(reward))

        #Critic update
        tdError = reward - self.averageReward + numpy.inner(newState.X, self.valueWeights) - numpy.inner(previousState.X, self.valueWeights)
        print("tdError: " + str(tdError))
        self.averageReward = self.averageReward + self.rewardStep * tdError
        print("Average reward: " + str(self.averageReward))
        self.elibibilityTraceValue = self.lambdaValue * self.elibibilityTraceValue + previousState.X
        self.valueWeights = self.valueWeights + self.stepSizeValue * tdError * self.elibibilityTraceValue

        m = self.mean(previousState)
        v = self.variance(previousState)

        #Mean Update
        self.elibibilityTraceMean = self.lambdaPolicy * self.elibibilityTraceMean + ((action - m) * previousState.X)
        self.policyWeightsMean = self.policyWeightsMean + self.stepSizeMean * tdError * self.elibibilityTraceMean

        #Variance Update
        logPie = (numpy.power(action - m, 2) - numpy.power(v, 2)) * previousState.X
        self.elibibilityTraceVariance = self.lambdaPolicy * self.elibibilityTraceVariance + logPie
        self.policyWeightsVariance = self.policyWeightsVariance + self.stepSizeVariance * tdError * self.elibibilityTraceVariance

        if reward == 1:
            print("logPie: " + str(logPie))
            print("Elg trace variance: ")
            print(self.elibibilityTraceVariance)
            print("policyWeightsVariance: " )
            print(self.policyWeightsVariance)

        pubReward = rospy.Publisher('horde_AC/Continuous/Reward', Float64, queue_size=10)
        pubReward.publish(reward)

        pubTD = rospy.Publisher('horde_AC/Continuous/TDError', Float64, queue_size=10)
        pubTD.publish(tdError)

        pubEncoder = rospy.Publisher('horde_AC/Continuous/Encoder', Float64, queue_size=10)
        pubEncoder.publish(newState.encoder / 100.0)

        pubAvgReward = rospy.Publisher('horde_AC/Continuous/AvgReward', Float64, queue_size=10)
        pubAvgReward.publish(self.averageReward)
        print("============ End continuous actor critic learn ============")
        print("-")

