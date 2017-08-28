"""
TD lambda code. Things to note:
 - Commented out "self.lastAction" since it doesn't seem to be doing anything. We can remove it for sure after we make sure it runs.
 - lambda, gamma, cumulant are hardcoded. We should pass them to the init.
 - Potentially we could switch to logging rather than printing 
"""

import numpy

class TDLambda:
    def __init__(self, featureVectorLength, alpha):
        #set up lambda, gamma, etc.
        self.numberOfFeatures = featureVectorLength
        self.lastState = 0
        self.lastObservation = 0
        self.weights = numpy.zeros(self.numberOfFeatures)
        self.eligibilityTrace = numpy.zeros(self.numberOfFeatures)
        self.gammaLast = 1

        self.alpha = alpha
        #self.lastAction = 0
        self.priorObservation = -1

    def lam(self, state, observation):
        return 0.95

    def gamma(self, state, observation):
        if (observation == -1):
            return 0

    def cumulant(self, state, observation):
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
        lam = self.lam(newState, observation)
        print("gammaLast: " + str(self.gammaLast))

        print("lambda: " + str(lam))

        self.eligibilityTrace = self.gammaLast * lam * self.eligibilityTrace + lastState

        tdError = zNext + gammaNext * numpy.inner(newState, self.weights) - numpy.inner(lastState, self.weights)
        print("tdError: " + str(tdError))

        self.weights = self.weights + self.alpha * tdError * self.eligibilityTrace

        pred = self.prediction(lastState)
        print("---Prediction after learning: " + str(pred))
        self.gammaLast = gammaNext
        self.priorObservation = observation

    def prediction(self, state):
        return numpy.inner(self.weights, state)
