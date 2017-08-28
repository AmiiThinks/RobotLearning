"""
Code for verifying some hyperparameters are changing as expected. We will need to change this slightly.
"""

import rospy

from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import String
from horde.msg import StateRepresentation

class Verifier:
    def __init__(self, bufferLength, name):
        self.bufferLength = bufferLength
        self.name = name
        self.gammas = []
        self.cumulants = []
        self.predictions = []
        self.observations = []

    def append(self, gamma, cumulant, prediction, newState):
        encoder_position = newState.encoder
        speed = newState.speed
        load = newState.load


        print("Verify append with observation: " + str(encoder_position) + ", speed: " + str(speed) + ", load: " + str(load) + ", gamma: " + str(gamma) + ", cumulant: " + str(cumulant) + ", prediction: " + str(prediction))
        if (len(self.gammas) == self.bufferLength):
            #Remove the first item from the arrays
            self.gammas.pop(0)
            self.cumulants.pop(0)
            self.predictions.pop(0)
            self.observations.pop(0)

        self.gammas.append(gamma)
        self.cumulants.append(cumulant)
        self.predictions.append(prediction)
        self.observations.append(encoder_position)
        #First gamma has to be 1 since the return of first is always just the unweighted cumulant

        runningGamma = 1
        runningCumulant = 0
        self.gammas[0] = 1
        for i in range(0, len(self.gammas)):
            runningGamma = runningGamma * self.gammas[i]
            runningCumulant = runningCumulant + self.cumulants[i] * runningGamma


            #TODO jump out of for loop if runningGamma = 0 since no point


        predict = self.predictions[0]
        #print("Observation " + str(self.observations[0]) +  ":")
        #print("--- Prediction: " + str(self.predictions[0]))

        #print("--- Actual: " + str(runningCumulant))

        #Publish the values

        pubPrediction = rospy.Publisher('horde_verifier/' + self.name + '/predicted', Float64, queue_size=10)
        pubPrediction.publish(self.predictions[0])

        pubActual = rospy.Publisher('horde_verifier/' + self.name + '/actual', Float64, queue_size=10)
        pubActual.publish(runningCumulant)

        pubError = rospy.Publisher('horde_verifier/' + self.name + '/error', Float64, queue_size=10)
        pubError.publish(self.predictions[0] - runningCumulant)

        pubObs = rospy.Publisher('horde_verifier/' + self.name + '/encoder_position', Int16, queue_size=10)
        pubObs.publish(self.observations[0] / 100)

