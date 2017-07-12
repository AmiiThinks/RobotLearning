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

        self.td_error = self.learner.delta
        self.avg_td_error = 0

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
        self.td_error = self.learner.delta
        if self.td_error > 0.001:
            if self.avg_td_error == 0:
                self.avg_td_error = self.td_error
            self.avg_td_error += 0.2 * (self.td_error - self.avg_td_error)

