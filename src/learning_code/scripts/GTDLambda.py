"""
Code for GTD lambda. Things to note:
 - rho needs to be passed in init.
 - could log rather than print.
 - Need clarification on the function of priorObservation.
 - normalizedObs is normalizing using hardcoded values.
"""

import numpy
import rospy
from std_msgs.msg import Float64
from TDLambda import TDLambda

class GTDLambda(TDLambda):
    def __init__(self, featureVectorLength, alpha):
        TDLambda.__init__(self, featureVectorLength, alpha)
        self.hWeights = numpy.zeros(featureVectorLength)
        self.priorObservation = -1 #Used only for debugging

    def rho(self):
        #To be overwritten based on the target policy
        return 1

    def learn(self, lastState, action, newState, observation):

        print("")
        print("!!!!! LEARN  !!!!!!!")
        print("For " + str(self.priorObservation) + " to " + str(observation))
        pred = self.prediction(lastState)
        print("--- Prediction before learning: " + str(pred))
        print("alpha: " + str(self.alpha))

        print("action")
        print(action)

        print("Observation:" + str(observation))

        zNext = self.cumulant(newState, observation)
        print("Cumulant: " + str(zNext))
        gammaNext = self.gamma(newState, observation)
        print("gammaNext: " + str(gammaNext))
        print("gammaLast: " + str(self.gammaLast))

        lam = self.lam(newState, observation)
        print("lambda: " + str(lam))

        rho = self.rho(action)
        print("rho: " + str(rho))
        self.eligibilityTrace = rho * (self.gammaLast * lam * self.eligibilityTrace + lastState)

        tdError = zNext + gammaNext * numpy.inner(newState, self.weights) - numpy.inner(lastState, self.weights)
        print("tdError: " + str(tdError))

        self.hWeights = self.hWeights + self.alpha * 0.1 * (tdError * self.eligibilityTrace - (numpy.inner(self.hWeights, lastState)) * lastState)
        self.weights = self.weights + self.alpha * (tdError * self.eligibilityTrace - gammaNext * (1-lam)  * (numpy.inner(self.eligibilityTrace, self.hWeights) * newState))

         pred = self.prediction(lastState)
        print("Prediction after learning: " + str(pred))

        self.gammaLast = gammaNext
        if (self.priorObservation > -1):
            pubPrediction = rospy.Publisher('horde_verifier/GTDpredictedValue', Float64, queue_size=10)
            pubPrediction.publish(pred)
            pubObs = rospy.Publisher('horde_verifier/NormalizedEncoderPosition', Float64, queue_size=10)
            normalizedObs = 3.0 * (self.priorObservation['encoder'] - 510.0) / (1023.0-510.0)
            pubObs.publish(normalizedObs )

        self.priorObservation = observation
