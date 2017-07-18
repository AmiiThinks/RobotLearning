import numpy as np
import rospy

class GTD:

	def __init__(self, parameters, num_features):
		self.num_features = num_features
		self.theta 		  = np.zeros(self.num_features)
		self.w			  = np.zeros(self.num_features)
		self.e     		  = np.zeros(self.num_features)
		
		self.alpha = parameters["alpha"] / self.num_features
		self.beta = parameters["beta"] / self.num_features
		self.lmbda = parameters["lambda"]
		self.old_gamma = 0
		self.delta = 0
		self.tderr_elig = 0

	def update(self, phi, phi_prime, cumulant, gamma, rho):
		self.tderr_elig = self.delta * self.e

		self.delta = cumulant + gamma * np.dot(phi_prime, self.theta) - np.dot(phi, self.theta)
		self.e = rho * (self.lmbda * self.old_gamma * self.e + phi)

		self.theta += self.alpha * (self.tderr_elig - gamma * (1 - self.lmbda) * np.dot(self.e, self.w) * phi_prime)
		self.w += self.beta * (self.tderr_elig - np.dot(phi, self.w) * phi)

		self.old_gamma = gamma
		return self.delta, self.e

	def predict(self, phi):
		return np.dot(phi, self.theta)
