import numpy as np

import tools


class GTD:
    """Implements GTD(lambda) with linear function approximation.

    Args:
        num_features (int): Length of weight vectors.
        alpha (float): Primary learning rate.
        beta (float): Secondary learning rate.
        lmbda (float): Trace decay rate.
        decay (bool, optional): Whether to decay alpha and beta.

    Attributes:
        theta: Primary weight vector.
        w: Secondary weight vector.
        e: Eligibility trace vector.
        alpha: Primary learning rate.
        beta: Secondary learning rate.
        lmbda: Trace decay rate.
        old_gamma: Discounting parameter from the previous timestep.
        delta: TD-error of previous timestep.
        tderr_elig: delta * e for RUPEE calculations.
    """

    def __init__(self,
                 num_features,
                 alpha,
                 beta,
                 lmbda,
                 decay=False,
                 **kwargs):
        self.theta = np.zeros(num_features)
        self.w = np.zeros(num_features)
        self.e = np.zeros(num_features)

        self.alpha = tools.decay(alpha) if decay else tools.constant(alpha)
        self.beta = tools.decay(beta) if decay else tools.constant(beta)
        self.lmbda = lmbda
        self.old_gamma = 0
        self.delta = 0
        self.tderr_elig = np.zeros(num_features)

    def update(self, phi, phi_prime, cumulant, gamma, rho, **kwargs):
        self.delta = (cumulant + gamma * np.dot(phi_prime, self.theta) -
                      np.dot(phi, self.theta))
        self.e = rho * (self.lmbda * self.old_gamma * self.e + phi)
        self.tderr_elig = self.delta * self.e

        self.theta += self.alpha.next() * (self.tderr_elig -
                                           gamma * (1 - self.lmbda) *
                                           np.dot(self.e, self.w) * phi_prime)
        self.w += self.beta.next() * (self.tderr_elig -
                                      np.dot(phi, self.w) * phi)

        self.old_gamma = gamma

        # for compatibility with calculating RUPEE for control gvfs
        return phi

    def predict(self, phi):
        return np.dot(phi, self.theta)
