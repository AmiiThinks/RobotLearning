import numpy as np


class WISGTD:
    """Implements WIS-GTD(lambda) with linear function approximation.

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
        self.u = np.ones(num_features) * u
        self.v = np.zeros(num_features)
        self.w = np.zeros(num_features)

        assert beta > 0 and eta > 0 and u > 0

        self.beta = beta
        self.eta = eta
        self.old_lmbda = lmbda
        self.old_gamma = 0
        self.delta = 0
        self.tderr_elig = np.zeros(num_features)

    def update(self, phi, phi_prime, cumulant, gamma, rho, **kwargs):

        lmbda = self.old_lmbda # replace this when lambda changes by state
        gam_lam = self.old_lmbda * self.old_gamma

        phi_sq = phi * phi
        k = np.ones(phi.size) - self.eta * phi_sq
        self.u *= k
        self.u += rho * phi_sq + (rho - 1) * gam_lam * k * self.v

        self.v *= gam_lam * rho * k
        self.v += rho * phi_sq

        non_zero = self.u != 0
        alpha = np.ones(self.u.size)[non_zero]/self.u[non_zero]
        alpha[~non_zero] = 0

        self.delta = (cumulant + gamma * np.dot(phi_prime, self.theta) -
                      np.dot(phi, self.theta))
        self.e = rho * (gam_lam * self.e + phi)
        self.tderr_elig = self.delta * self.e

        self.theta += alpha * (self.tderr_elig - (gamma * (1 - lmbda) *
                                                  np.dot(self.e, self.w) *
                                                  phi_prime))
        self.w += self.beta * (self.tderr_elig - np.dot(self.w, phi) * phi)

        self.old_gamma = gamma
        self.old_lmbda = lmbda

        # for compatibility with calculating RUPEE for control gvfs
        return phi

    def predict(self, phi):
        return np.dot(phi, self.theta)
