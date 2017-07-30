"""
Author: Banafsheh Rafiee

"""
import numpy as np
from state_representation import StateConstants
from evaluator import Evaluator

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

        self.time_step = 0

        # See Adam White's PhD Thesis, section 8.4.2
        self.alpha_rupee = 5 * parameters['alpha']
        self.beta0_rupee = (1 - parameters['lambda'])*parameters['alpha0']/30
        self.evaluator = Evaluator(gvf_name = name, 
                                   num_features = num_features, 
                                   alpha_rupee = self.alpha_rupee, 
                                   beta0_rupee = self.beta0_rupee)
    
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
        self.evaluator.compute_rupee(tderr_elig = self.learner.tderr_elig, 
                                     delta = self.learner.delta,
                                     phi = phi)
        # update MSRE
        # self.evaluator.compute_MSRE(self.learner.theta)

        # update avg TD error
        self.evaluator.compute_avg_td_error(delta = self.learner.delta, 
                                            time_step = self.time_step)
        self.phi = phi_prime
        self.time_step = self.time_step + 1
 