"""
GVF code. Things to note:
 - alpha is scaled by 0.1
 - hardcoded gamma, cumulant, lambda, rho. These things can be overwritten but it's not clear how. Perhaps the import * are hiding the trick.
 - This file includes TD code. Is it possible to use TDLambda.py instead? 
"""

import numpy
import rospy
from std_msgs.msg import Float64
from horde.msg import StateRepresentation
from TileCoder import *

class GVF:
    def __init__(self, featureVectorLength, alpha, isOffPolicy, name = "GVF name"):
        #set up lambda, gamma, etc.
        self.name = name
        self.isOffPolicy = isOffPolicy
        self.numberOfFeatures = featureVectorLength
        self.lastState = 0
        self.lastObservation = 0
        self.weights = numpy.zeros(self.numberOfFeatures)
        self.hWeights = numpy.zeros(featureVectorLength)
        self.hHatWeights = numpy.zeros(featureVectorLength)
        self.eligibilityTrace = numpy.zeros(self.numberOfFeatures)
        self.gammaLast = 1

        self.alpha = (1.0 - 0.90) * alpha
        self.alphaH = 0.01 * self.alpha #Same alpha for H vector and each HHat vector

        self.alphaRUPEE = 5.0 * self.alpha
        self.betaNotUDE = self.alpha * 10.0
        self.betaNotRUPEE = (1.0 - 0.90) * alpha * TileCoder.numberOfTilings / 30
        self.taoRUPEE = 0
        self.taoUDE = 0
        self.movingtdEligErrorAverage = 0 #average of TD*elig*hHat
        self.lastAction = 0

        self.tdVariance = 0
        self.averageTD = 0
        self.i = 1

    """
    gamma, cumulant, and policy functions can/should be overiden by the specific instantiation of the GVF based on the intended usage.
    """
    def gamma(self, state):
        return 0.0

    def cumulant(self, state):
        return 1

    def policy(self, state):
        #To be overwritten based on GVF's intended behavior if off policy. Otherwise 1 means on policy
        return 1

    def lam(self, state):
        return 0.90

    def rho(self, action, state):
        targetAction = self.policy(state)
        if (targetAction == action):
            return 1
        else:
            return 0

    def learn(self, lastState, action, newState):
        if self.isOffPolicy:
            self.gtdLearn(lastState, action, newState)
        else:
            self.tdLearn(lastState, action, newState)

    def gtdLearn(self, lastState, action, newState):

        #print("")
        print("!!!!! LEARN  !!!!!!!")
        print("GVF name: " + str(self.name))
        print("For (" + str(lastState.encoder) +  ", " + str(lastState.speed) +  ") to (" + str(newState.encoder) + ", " + str(newState.speed) + ")")
        pred = self.prediction(lastState)
        #print("--- Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed) + " before learning: " + str(pred))

        #print("action")
        #print(action)
        zNext = self.cumulant(newState)
        #print("Cumulant: " + str(zNext))
        gammaNext = self.gamma(newState)
        #print("gammaNext: " + str(gammaNext))
        lam = self.lam(newState)
        #print("gammaLast: " + str(self.gammaLast))

        #print("lambda: " + str(lam))
        rho = self.rho(action, lastState)
        #print("rho: " + str(rho))
        self.eligibilityTrace = rho * (self.gammaLast * lam * self.eligibilityTrace + lastState.X)
        tdError = zNext + gammaNext * numpy.inner(newState.X, self.weights) - numpy.inner(lastState.X, self.weights)


        #print("tdError: " + str(tdError))


        self.hWeights = self.hWeights + self.alphaH  * (tdError * self.eligibilityTrace - (numpy.inner(self.hWeights, lastState.X)) * lastState.X)

        #update Rupee
        self.hHatWeights = self.hHatWeights + self.alphaRUPEE * (tdError * self.eligibilityTrace - (numpy.inner(self.hHatWeights, lastState.X)) * lastState.X)
        #print("tao before: " + str(self.tao))
        self.taoRUPEE = (1.0 - self.betaNotRUPEE) * self.taoRUPEE + self.betaNotRUPEE
        #print("tao after: " + str(self.tao))

        betaRUPEE = self.betaNotRUPEE / self.taoRUPEE
        #print("beta: " + str(beta))
        self.movingtdEligErrorAverage = (1.0 - betaRUPEE) * self.movingtdEligErrorAverage + betaRUPEE * tdError * self.eligibilityTrace

        #update UDE
        self.taoUDE = (1.0 - self.betaNotUDE) * self.taoUDE + self.betaNotUDE
        betaUDE = self.betaNotUDE / self.taoUDE

        oldAverageTD = self.averageTD
        #print("Old averageTD:" + str(oldAverageTD))


        self.averageTD = (1.0 - betaUDE) * self.averageTD + betaUDE * tdError
        #print("New AverageTD: " + str(self.averageTD))
        #print("tdvariance before: " + str(self.tdVariance))
        self.tdVariance = ((self.i - 1) * self.tdVariance + (tdError - oldAverageTD) * (tdError - self.averageTD)) / self.i
        #print("td variance after: " + str(self.tdVariance))
        self.i = self.i + 1

        self.weights = self.weights + self.alpha * (tdError * self.eligibilityTrace - gammaNext * (1-lam)  * (numpy.inner(self.eligibilityTrace, self.hWeights) * newState.X))

        pred = self.prediction(lastState)
        #print("Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed)  + " after learning: " + str(pred))

        rupee = self.rupee()
        #print("Rupee: " + str(rupee))

        ude = self.ude()
        #print("UDE: " + str(ude))

        self.gammaLast = gammaNext


    def tdLearn(self, lastState, action, newState):
        print("!!!!! LEARN  !!!!!!!")
        print("GVF name: " + str(self.name))
        print("For (" + str(lastState.encoder) +  ", " + str(lastState.speed) +  ") to (" + str(newState.encoder) + ", " + str(newState.speed) + ")")
        pred = self.prediction(lastState)
        print("--- Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed) + " before learning: " + str(pred))

        #print("alpha: " + str(self.alpha))

        #print("action")
        #print(action)

        zNext = self.cumulant(newState)
        #print("Cumulant: " + str(zNext))
        gammaNext = self.gamma(newState)
        #print("gammaNext: " + str(gammaNext))
        lam = self.lam(newState)
        #print("gammaLast: " + str(self.gammaLast))

        #print("lambda: " + str(lam))
        self.eligibilityTrace = self.gammaLast * lam * self.eligibilityTrace + lastState.X

        tdError = zNext + gammaNext * numpy.inner(newState.X, self.weights) - numpy.inner(lastState.X, self.weights)

        #print("tdError: " + str(tdError))

        #update Rupee
        self.hHatWeights = self.hHatWeights + self.alphaRUPEE * (tdError * self.eligibilityTrace - (numpy.inner(self.hHatWeights, lastState.X)) * lastState.X)
        #print("tao before: " + str(self.tao))
        self.taoRUPEE = (1.0 - self.betaNotRUPEE) * self.taoRUPEE + self.betaNotRUPEE
        #print("tao after: " + str(self.taoRUPEE))

        betaRUPEE = self.betaNotRUPEE / self.taoRUPEE
        #print("beta: " + str(beta))
        self.movingtdEligErrorAverage =(1.0 - betaRUPEE) * self.movingtdEligErrorAverage + betaRUPEE * tdError * self.eligibilityTrace


        #update UDE
        self.taoUDE = (1.0 - self.betaNotUDE) * self.taoUDE + self.betaNotUDE
        betaUDE = self.betaNotUDE / self.taoUDE
        oldAverageTD = self.averageTD
        self.averageTD = (1.0 - betaUDE) * self.averageTD + betaUDE * tdError
        self.tdVariance = ((self.i - 1) * self.tdVariance + (tdError - oldAverageTD) * (tdError - self.averageTD)) / self.i
        self.i = self.i + 1

        self.weights = self.weights + self.alpha * tdError * self.eligibilityTrace

        pred = self.prediction(lastState)
        print("Prediction for " + str(lastState.encoder) + ", " + str(lastState.speed)  + " after learning: " + str(pred))
        rupee = self.rupee()

        #print("Rupee: " + str(rupee))

        ude = self.ude()
        #print("UDE: " + str(ude))

        self.gammaLast = gammaNext

    def prediction(self, stateRepresentation):
        return numpy.inner(self.weights, stateRepresentation.X)

    def rupee(self):
        return numpy.sqrt(numpy.absolute(numpy.inner(self.hHatWeights, self.movingtdEligErrorAverage)))

    def ude(self):
        return numpy.absolute(self.averageTD / (numpy.sqrt(self.tdVariance) + 0.000001))
