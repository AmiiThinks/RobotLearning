"""
Author: Banafsheh Rafiee, Niko Yasui
"""
from __future__ import division

import numpy as np

from evaluator import Evaluator


class GVF:
    def __init__(self,
                 cumulant,
                 gamma,
                 target_policy,
                 num_features,
                 alpha0,
                 alpha,
                 name,
                 learner,
                 feature_indices,
                 use_MSRE=False,
                 **kwargs):

        self.cumulant = cumulant
        self.gamma = gamma
        self.target_policy = target_policy
        self.last_cumulant = 0.0
        self.phi = np.zeros(num_features)
        self.rho = 1.0
        self.last_prediction = 0.0

        self.name = name
        self.feature_indices = feature_indices
        self.learner = learner
        self.uses_action_state = feature_indices.size < num_features
        self.use_MSRE = use_MSRE

        self.time_step = 0

        # See Adam White's PhD Thesis, section 8.4.2
        self.alpha_rupee = 5 * alpha
        self.beta0_rupee = alpha0 / 30
        self.evaluator = Evaluator(gvf_name=name,
                                   num_features=num_features,
                                   alpha_rupee=self.alpha_rupee,
                                   beta0_rupee=self.beta0_rupee,
                                   use_MSRE=use_MSRE)

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
               mu,
               action):

        self.last_prediction = self.predict(phi_prime, action)

        # update action probabilities and get probability of last action
        self.target_policy.update(phi, last_observation)
        pi = self.target_policy.get_probability(last_action)

        self.last_cumulant = self.cumulant(last_observation, phi)

        # get relevant indices in phi
        phi = phi[self.feature_indices]
        phi_prime = phi_prime[self.feature_indices]

        self.rho = pi / mu
        kwargs = {"phi": phi,
                  "last_action": last_action,
                  "phi_prime": phi_prime,
                  "rho": self.rho,
                  "gamma": self.gamma(observation),
                  "cumulant": self.last_cumulant,
                  }

        phi = self.learner.update(**kwargs)

        self.evaluator.update(theta=self.learner.theta,
                              time_step=self.time_step,
                              tderr_elig=self.learner.tderr_elig,
                              delta=self.learner.delta,
                              phi=phi,
                              rho=self.rho)

        self.phi = phi_prime
        self.time_step += 1
