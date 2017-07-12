"""
Author: Banafsheh Rafiee

"""
import numpy as np

class GVF:
    def __init__(self, 
                 cumulant, 
                 gamma, 
                 target_policy, 
                 num_features,
                 parameters,
                 off_policy, 
                 alg,
                 name, 
                 logger):

        self.cumulant = cumulant
        self.gamma = gamma
        self.target_policy = target_policy

        self.parameters = parameters
        self.num_features = num_features
        self.phi = np.zeros(num_features)
        
        self.off_policy = off_policy
        self.name = name

        self.learner = alg(parameters, self.num_features)

        self.predict = self.learner.predict

    def update(self, 
               last_observation, 
               phi, 
               last_action, 
               observation, 
               phi_prime, 
               mu):
        pi = self.target_policy(last_observation, last_action)[1]
        self.learner.update(phi = phi,
                            phi_prime = phi_prime,
                            cumulant = self.cumulant(observation),
                            gamma = self.gamma(observation),
                            rho = pi / mu) 
        self.phi = phi_prime

