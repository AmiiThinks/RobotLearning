"""
Authors:
    Banafsheh Rafiee, Niko Yasui
"""
from __future__ import division

import numpy as np

from evaluator import Evaluator


class GVF:
    """Implements General Value Functions.

    General Value Functions pose a question defined by the cumulant, gamma,
    and the target policy, that is learned by a learning algorithm, here called
    the ``learner``.

    Args:
        cumulant (fun): Function of observation that gives a float value.
        gamma (fun): Function of observation that gives a float value. Together
            with cumulant, makes the return that the agent tries to predict.
        target_policy (Policy): Policy under which the agent makes its
            predictions. Can be the same as the behavior policy.
        num_features (int): Number of features that are used.
        alpha0 (float): Value to calculate beta0 for RUPEE.
        alpha (float): Value to calculate alpha for RUPEE.
        name (str): Name of the GVF for recording data.
        learner: Class instance with a ``predict`` and ``update`` function,
            and ``theta``, ``tderr_elig``, and ``delta`` attributes. For
            example, GTD.
        feature_indices (numpy array of bool): Indices of the features to use.
        use_MSRE (bool): Whether or not to calculate MSRE.
    """
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
        alpha_rupee = 5 * alpha
        beta0_rupee = alpha0 / 30
        self.evaluator = Evaluator(gvf_name=name,
                                   num_features=num_features,
                                   alpha_rupee=alpha_rupee,
                                   beta0_rupee=beta0_rupee,
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

        cumulant = self.cumulant(observation)

        # get relevant indices in phi
        phi = phi[self.feature_indices]
        phi_prime = phi_prime[self.feature_indices]

        self.rho = pi / mu
        kwargs = {"phi": phi,
                  "last_action": last_action,
                  "phi_prime": phi_prime,
                  "rho": self.rho,
                  "gamma": self.gamma(observation),
                  "cumulant": cumulant,
                  }

        phi = self.learner.update(**kwargs)

        self.evaluator.update(theta=self.learner.theta,
                              time_step=self.time_step,
                              tderr_elig=self.learner.tderr_elig,
                              delta=self.learner.delta,
                              phi=phi,
                              rho=self.rho)

        self.phi = phi_prime
        self.last_cumulant = cumulant
        self.time_step += 1
