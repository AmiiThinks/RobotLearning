import numpy as np
import rospy

class GTD:

    def __init__(self, num_features, alpha, beta, lmbda, **kwargs):
        self.theta        = np.zeros(num_features)
        self.w            = np.zeros(num_features)
        self.e            = np.zeros(num_features)
        
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda
        self.old_gamma = 0
        self.delta = 0
        self.tderr_elig = np.zeros(num_features)

    def update(self, phi, phi_prime, cumulant, gamma, rho, **kwargs):
        self.delta = cumulant + gamma * np.dot(phi_prime, self.theta) - np.dot(phi, self.theta)
        self.e = rho * (self.lmbda * self.old_gamma * self.e + phi)
        self.tderr_elig = self.delta * self.e
        
        self.theta += self.alpha * (self.tderr_elig - gamma * (1 - self.lmbda) * np.dot(self.e, self.w) * phi_prime)
        self.w += self.beta * (self.tderr_elig - np.dot(phi, self.w) * phi)

        self.old_gamma = gamma

        # for compatibility with calculating RUPEE for control gvfs
        return phi

    def predict(self, phi):
        return np.dot(phi, self.theta)
