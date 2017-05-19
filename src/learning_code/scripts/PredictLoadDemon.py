"""
Note: We need a similar file to make our demons.
"""

from TDLambda import *

class PredictLoadDemon(TDLambda):
    def __init__(self, featureVectorLength, alpha):
        TDLambda.__init__(self, featureVectorLength, alpha)

    def gamma(self, state, observation):
        return 0.5

    def cumulant(self, state, observation):
        return observation['load']
