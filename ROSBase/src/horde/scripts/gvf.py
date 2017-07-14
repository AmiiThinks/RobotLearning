"""
Author: David Quail, Niko Yasui, June, 2017.

Description:
The GVF base class allows users to instantiate a specific general value
function. The GVF could answer any specific question by overwriting the 
behavior policy, cumulant function, or gamma.

"""
from __future__ import print_function
import numpy as np

from algorithms import GTD
from policy import Policy
import random

class GVF:
    def __init__(self, 
                 num_features, 
                 alpha, 
                 beta,
                 lambda_=lambda observation: 0.9,
                 gamma=lambda observation: 0,
                 cumulant=lambda observation: 1,
                 policy=Policy(),
                 off_policy=True, 
                 alg=GTD,
                 name='GVFname', 
                 logger=print):

        self.lambda_ = lambda_
        self.gamma = gamma
        self.cumulant = cumulant
        self.policy = policy
        self.rho = lambda action, observation, mu, phi: policy.prob(action, phi, observation) / mu
        self.off_policy = off_policy
        self.name = name

        theta = np.random.rand(num_features)
        phi = np.zeros(num_features)
        observation = None
        learningRate = 0.1
        secondaryLearningRate = 0.1
        epsilon = 0.5
        lambda_ = 0.5

        self.learner = alg(theta = theta,
                        gamma = gamma,
                        _lambda = lambda_,
                        cumulant = cumulant,
                        alpha=learningRate,
                        beta=secondaryLearningRate,
                        epsilon=epsilon)

        if False:
            self.predict = self.learner.predict

    def update(self, 
               last_action, 
               phi_prime, 
               new_observation, 
               last_observation, 
               last_mu):
        if self.off_policy:
            self.learner.update(phi_prime, 
                                self.cumulant(new_observation),
                                self.rho(last_action, 
                                         last_observation,
                                         last_mu,
                                         phi_prime),
                                lambda_=self.lambda_(new_observation),
                                gamma=self.gamma(new_observation))
        else:
            self.learner.update(phi_prime, 
                                self.cumulant(new_observation),
                                1,
                                lambda_=self.lambda_(new_observation),
                                gamma=self.gamma(new_observation))


