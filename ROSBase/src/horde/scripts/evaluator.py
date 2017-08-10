"""
Author: Banafsheh Rafiee, Niko Yasui

"""
import numpy as np

class Evaluator:
    def __init__(self,
                 gvf_name,
                 num_features,
                 alpha_rupee,
                 beta0_rupee,
                 use_MSRE):

        MAX_TIME_STEPS = 1000000

        # load the state representation and actual return for sample states
        if use_MSRE:
            data = np.load("actual_return_" + gvf_name + ".npz")
            self.samples_phi = data["samples"]
            self.samples_G   = data["_return"]
            self.sample_size = data["sample_size"]

        # # initialize the preformance measures
        self.MSRE = 0.0
        self.MSRE_over_time = np.zeros(MAX_TIME_STEPS)
        self.td_error = 0.0
        self.avg_td_error = 0.0
        
        # See Adam White's PhD Thesis, section 8.4.2
        self.alpha_rupee = alpha_rupee
        self.beta0_rupee = beta0_rupee
        self.tau_rupee = 0.0
        self.hhat = np.zeros(num_features)
        self.td_elig_avg = np.zeros(num_features)
        self.rupee = 0.0

    def compute_MSRE(self, theta, time_step):
        return_error = 0.0
        for i, phi in enumerate(self.samples_phi):
            estimated_value = np.dot(theta, phi)
            return_error = return_error + (estimated_value - self.samples_G[i]) * (estimated_value - self.samples_G[i])
        MSRE = np.sqrt(return_error / self.sample_size)
        self.MSRE = MSRE
        self.MSRE_over_time[time_step] = MSRE
        if (time_step % 10 == 0.0):
            np.savez("MSRE_over_time.npz", MSRE = self.MSRE_over_time, time_step = time_step) 

    def compute_rupee(self, tderr_elig, delta, phi):

        # update RUPEE
        # add condition to change for control gvf
        self.hhat += self.alpha_rupee * (tderr_elig - np.inner(self.hhat, phi) * phi)
        self.tau_rupee *= 1 - self.beta0_rupee
        self.tau_rupee += self.beta0_rupee
        beta_rupee = self.beta0_rupee / self.tau_rupee
        self.td_elig_avg *= 1 - beta_rupee
        self.td_elig_avg += beta_rupee * tderr_elig
 
        self.rupee = np.sqrt(np.absolute(np.inner(self.hhat, self.td_elig_avg)))

    def compute_avg_td_error(self, delta, time_step):
        self.avg_td_error += (delta - self.avg_td_error)/(time_step + 1)
        self.td_error = delta