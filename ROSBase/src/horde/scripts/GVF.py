"""
Author: David Quail, June, 2017.

Description:
The GVF base class allows users to instantiate a specific general value function. The GVF could answer any specific question
 by overwritingn the behavior policy, cumulant function, or gamma.

 Reference:
 See Adam White's thesis for refernce on GTD, TD, RUPEE and UDE algorithms
 http://homes.soic.indiana.edu/adamw/phd.pdf


Examples:

1. Create a GVF that predicts the value of a certain bit in the feature vector

#For this we create a helper function that will dynamically create cumulant function for us. This way, we could
#create a cumulant function for all 1024 bits in the vector using a for loop

def makeVectorBitCumulantFunction(bitIndex):
    def cumulantFunction(state):
        if (state.X[bitIndex] == 1):
            return 1
        else:
            return 0
    return cumulantFunction

    gvfs = []

    gvfOn = GVF(TileCoder.numberOfTiles * TileCoder.numberOfTiles * TileCoder.numberOfTilings,
                0.1 / TileCoder.numberOfTilings, isOffPolicy=False, name="NextBitOnPolicy" + str(301))
    gvfOn.cumulant = makeVectorBitCumulantFunction(301)
    gvfs.append(gvfOn)


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
"""

import numpy

class GVF:
    def __init__(self, featureVectorLength, alpha, isOffPolicy, name = "GVF name"):

        self.name = name #Can be useful for debugging, logging, plotting. But not needed otherwise.
        self.isOffPolicy = isOffPolicy #Determines whether td or gtdlearn is used for learning
        self.numberOfFeatures = featureVectorLength
        self.lastState = 0
        self.lastObservation = 0
        self.weights = numpy.zeros(self.numberOfFeatures)
        self.hWeights = numpy.zeros(featureVectorLength) #For GTD off policy learning
        self.hHatWeights = numpy.zeros(featureVectorLength) #For RUPEE error estimation
        self.eligibilityTrace = numpy.zeros(self.numberOfFeatures)
        self.gammaLast = 1

        self.alpha = (1.0 - 0.90) * alpha
        self.alphaH = 0.01 * self.alpha #Same alpha for H vector and each HHat vector

        self.alphaRUPEE = 5.0 * self.alpha
        self.betaNotUDE = self.alpha * 10.0
        self.betaNotRUPEE = (1.0 - 0.90) * alpha * 6 / 30
        self.taoRUPEE = 0
        self.taoUDE = 0
        self.movingtdEligErrorAverage = 0 #average of TD*elig*hHat
        self.lastAction = 0

        self.tdVariance = 0
        self.averageTD = 0
        self.i = 1

    """
    GVF Question parameters
    - gamma, cumulant, and policy functions can/should be overiden by the specific instantiation of the GVF based on the
    intended question being asked.
    """
    def gamma(self, state):
        return 0.0

    def cumulant(self, state):
        return 1

    def policy(self, state):
        #To be overwritten based on GVF's intended behavior if off policy. Otherwise 1 means on policy
        return 1

    """
    Lambda value to control the extent to which updates are backed up
    """
    def lam(self, state):
        return 0.90

    """
    Simple, binary approach to determining whether updates should be applied or not.
    """
    def rho(self, action, state):
        targetAction = self.policy(state)
        if (targetAction == action):
            return 1
        else:
            return 0

    """
    Actual learning
    """
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

    """
    Prediction and error functions
    """
    def prediction(self, stateRepresentation):
        return numpy.inner(self.weights, stateRepresentation.X)

    def rupee(self):
        return numpy.sqrt(numpy.absolute(numpy.inner(self.hHatWeights, self.movingtdEligErrorAverage)))

    def ude(self):
        return numpy.absolute(self.averageTD / (numpy.sqrt(self.tdVariance) + 0.000001))