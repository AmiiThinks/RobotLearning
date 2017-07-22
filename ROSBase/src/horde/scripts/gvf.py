"""
Author: Banafsheh Rafiee

"""
import numpy as np
from state_representation import StateConstants

class GVF:
    def __init__(self, 
                 cumulant, 
                 gamma, 
                 target_policy, 
                 num_features,
                 parameters,
                 name, 
                 learner,
                 logger,
                 features_to_use):


        self.cumulant = cumulant
        self.last_cumulant = self.cumulant
        self.gamma = gamma
        self.target_policy = target_policy

        self.phi = np.zeros(num_features)
        
        self.name = name
        self.feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        self.learner = learner

        if self.learner is not None:
            self.td_error = self.learner.delta
            self.avg_td_error = 0
            self.n = 0

            # See Adam White's PhD Thesis, section 8.4.2
            self.alpha_rupee = 5 * parameters['alpha']
            self.beta0_rupee = (1 - parameters['lambda'])*parameters['alpha0']/30
            self.tau_rupee = 0
            self.hhat = np.zeros(num_features)
            self.td_elig_avg = np.zeros(num_features)

    def predict(self, phi):
        return self.learner.predict(phi[self.feature_indices])

    def update(self, 
               last_observation, 
               phi, 
               last_action, 
               observation, 
               phi_prime, 
               mu):

        pi = self.target_policy(phi, last_observation)[1]
        self.last_cumulant = self.cumulant(observation)

        phi_prime = phi_prime[self.feature_indices]
        phi = phi[self.feature_indices]

        kwargs = {"last_observation": last_observation,
                  "phi": phi,
                  "last_action": last_action,
                  "observation": observation,
                  "phi_prime": phi_prime,
                  "rho": pi / mu,
                  "gamma": self.gamma(observation),
                  "cumulant": self.last_cumulant,
                 }

        phi = self.learner.update(**kwargs)
        
        # update RUPEE
        # add condition to change for control gvf
        self.hhat += self.alpha_rupee * (self.learner.tderr_elig - np.inner(self.hhat, phi) * phi)
        self.tau_rupee *= 1 - self.beta0_rupee
        self.tau_rupee += self.beta0_rupee
        beta_rupee = self.beta0_rupee / self.tau_rupee
        self.td_elig_avg *= 1 - beta_rupee
        self.td_elig_avg += beta_rupee * self.learner.tderr_elig

        self.phi = phi_prime
        self.td_error = self.learner.delta
        self.avg_td_error += (self.td_error - self.avg_td_error)/(self.n + 1)
 
    def rupee(self):
        return np.sqrt(np.absolute(np.inner(self.hhat, self.td_elig_avg)))
