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
        self.n = 0

        # See Adam White's PhD Thesis, section 8.4.2
        self.alpha_rupee = 5 * parameters['alpha']
        self.beta0_rupee = (1 - parameters['lambda'])*parameters['alpha0']/30
        self.tau_rupee = 0
        self.hhat = np.zeros(num_features)
        self.td_elig_avg = np.zeros(num_features)



    def update(self, 
               last_observation, 
               phi, 
               last_action, 
               observation, 
               phi_prime, 
               mu):
        pi = self.target_policy(last_observation, last_action)[1]
        self.gamma_t = self.cumulant(observation)
        self.learner.update(phi = phi,
                            phi_prime = phi_prime,
                            cumulant = self.gamma_t,
                            gamma = self.gamma(observation),
                            rho = pi / mu) 
        
        # update RUPEE
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
