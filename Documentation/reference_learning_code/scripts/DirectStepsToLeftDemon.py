"""
Note: We need a similar file to make our demons.
"""

from GTDLambda import *
from TileCoder import *
import numpy

class DirectStepsToLeftDemon(GTDLambda):

    def __init__(self, featureVectorLength, alpha):
        GTDLambda.__init__(self, featureVectorLength, alpha)

    def gamma(self, state, observation):
        encoder = observation['encoder']
        if (encoder == -1):
            return 0
        elif (encoder == 1023):
            #This represents the extreme position
            return 0
        else:
            return 1

    def rho(self, action):
        if (action == 2):
            #our policy is to always move left.
            #ie. how many steps if we were to go directly to the left.
            return 1
        else:
            return 0

def test():
    d = DirectStepsToLeftDemon(8*8*8, 0.1/8)
    numTilings = 8
    numTiles = 8

    encoderPosition = 600
    speed = 100
    firstState =  firstState = tileCode(numTilings, numTilings * numTiles * numTiles, [((encoderPosition-510.0)/(1023.0-510.0)) * numTiles, ((speed + 200.0) / 400.0) * numTiles])

    encoderPosition = 1023
    speed = 0
    secondState =  firstState = tileCode(numTilings, numTilings * numTiles * numTiles, [((encoderPosition-510.0)/(1023.0-510.0)) * numTiles, ((speed + 200.0) / 400.0) * numTiles])

    d.learn(firstState, 2, secondState, 1023)

