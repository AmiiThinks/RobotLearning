import numpy as np


class WISTOGTD:
    """Implements WIS-TO-GTD(lambda) with linear function approximation.

    See https://armahmood.github.io/files/MS-WIS-O(n)-UAI-2015.pdf for more
    details.

    Args:
        num_features (int): Length of weight vectors.
        u (float): Initial value for the usage vector. Can be interpreted as
            inverse initial step size.
        eta (float): Recency-weighting factor. Can be interpreted as desired
            final step size.
        beta (float): Secondary learning rate.
        lmbda (float): Trace decay rate.

    Attributes:
        theta: Primary weight vector.
        w: Secondary weight vector.
        e: Eligibility trace vector.
        u: Usage vector.
        v: Usage helper vector.
        beta: Secondary learning rate.
        lmbda: Trace decay rate.
        old_gamma: Discounting parameter from the previous timestep.
        old_rho: Importance sampling weight from previous timestep.
        delta: TD-error of previous timestep.
        tderr_elig: delta * e for RUPEE calculations.
    """

    def __init__(self,
                 num_features,
                 u,
                 eta,
                 beta,
                 lmbda,
                 **kwargs):
        self.e = np.zeros(num_features)
        self.theta = np.zeros(num_features)
        self.old_theta = np.zeros(num_features)
        self.u = np.ones(num_features) * u
        self.v = np.zeros(num_features)
        self.w = np.zeros(num_features)
        self.e_grad = np.zeros(num_features)
        self.e_w = np.zeros(num_features)

        assert beta > 0 and eta > 0 and u > 0

        self.beta = beta
        self.eta = eta
        self.old_lmbda = lmbda
        self.old_gamma = 0
        self.delta = 0
        self.old_rho = 1
        self.tderr_elig = np.zeros(num_features)

    def update(self, phi, phi_prime, cumulant, gamma, rho, **kwargs):

        lmbda = self.old_lmbda # replace this when lambda changes by state
        gam_lam = self.old_lmbda * self.old_gamma
        temp = self.theta

        phi_sq = phi * phi
        k = np.ones(phi.size) - self.eta * phi_sq
        self.u *= k
        self.u += rho * phi_sq + (rho - 1) * gam_lam * k * self.v

        self.v *= gam_lam * rho * k
        self.v += rho * phi_sq

        non_zero = self.u != 0
        alpha = np.ones(self.u.size)[non_zero]/self.u[non_zero]
        alpha[~non_zero] = 0

        self.e = (rho * alpha * phi +
                  (gam_lam * rho * (self.e - rho * alpha * phi) *
                   np.dot(phi,self.e)))
        self.e_grad = rho * (gam_lam * self.e_grad + phi)
        self.e_w = (gam_lam * self.old_rho * self.e_w +
                    (self.beta *
                     (1 - gam_lam * self.old_rho * np.dot(phi, self.e_w)) *
                     phi))

        self.delta = (cumulant + gamma * np.dot(phi_prime, self.theta) -
                      np.dot(phi, self.theta))
        self.tderr_elig = self.delta * self.e

        self.theta += (self.tderr_elig
                       + ((self.e - alpha * rho * phi) *
                          (self.theta - self.old_theta) * phi)
                       - (alpha * gamma * (1 - lmbda) *
                          np.dot(self.w, self.e_grad)) * phi_prime)
        self.w += (rho * self.delta * self.e_w
                   - self.beta * np.dot(self.w, phi) * phi)

        self.old_gamma = gamma
        self.old_lmbda = lmbda
        self.old_rho = rho
        self.old_theta = temp

        # for compatibility with calculating RUPEE for control gvfs
        return phi

    def predict(self, phi):
        return np.dot(phi, self.theta)
