"""
Author: Banafsheh Rafiee

"""
from __future__ import division
import numpy as np
from state_representation import StateConstants

class GVF:
    def __init__(self, 
                 cumulant, 
                 gamma, 
                 target_policy, 
                 num_features,
                 alpha0,
                 alpha,
                 lmbda,
                 name, 
                 learner,
                 logger,
                 feature_indices,
                 **kwargs):

        self.cumulant = cumulant
        self.gamma = gamma
        self.target_policy = target_policy 
        
        self.name = name
        self.feature_indices = feature_indices
        self.learner = learner
        self.uses_action_state = feature_indices.size < num_features

        if self.learner is not None:
            self.td_error = self.learner.delta
            self.avg_td_error = 0
            self.n = 0
            self.last_cumulant = 0


            # See Adam White's PhD Thesis, section 8.4.2
            self.alpha_rupee = 5 * alpha
            self.beta0_rupee = alpha0 / 30
            self.tau_rupee = 0
            self.hhat = np.zeros(num_features)
            self.td_elig_avg = np.zeros(num_features)


    def predict(self, phi, action=None, **kwargs):
        if self.uses_action_state:
            return self.learner.predict(phi[self.feature_indices], action)
        else:
            return self.learner.predict(phi[self.feature_indices])

    def update(self, 
               last_observation, 
               phi, 
               last_action, 
               observation, 
               phi_prime, 
               mu):

        # update action probabilities and get probability of last action
        self.target_policy.update(phi, last_observation)
        pi = self.target_policy.get_probability(last_action)

        self.last_cumulant = self.cumulant(observation)

        # get relevant indices in phi
        phi = phi[self.feature_indices]
        phi_prime = phi_prime[self.feature_indices]

        kwargs = {"phi": phi,
                  "last_action": last_action,
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

        self.td_error = self.learner.delta
        self.avg_td_error += (self.td_error - self.avg_td_error)/(self.n + 1)
 
    def rupee(self):
        return np.sqrt(np.absolute(np.inner(self.hhat, self.td_elig_avg)))
