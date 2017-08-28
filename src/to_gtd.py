import numpy as np

import tools


class TOGTD:
    """Implements True Online GTD(lambda) with linear function approximation.

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
        e_grad: Gradient correction trace vector.
        e_w: Secondary eligibility trace vector.
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
        self.old_theta = np.zeros(num_features)
        self.w = np.zeros(num_features)
        self.e = np.zeros(num_features)
        self.e_grad = np.zeros(num_features)
        self.e_w = np.zeros(num_features)

        self.alpha = tools.decay(alpha) if decay else tools.constant(alpha)
        self.beta = tools.decay(beta) if decay else tools.constant(beta)
        self.lmbda = lmbda
        self.old_gamma = 0
        self.delta = 0
        self.old_rho = 1
        self.tderr_elig = np.zeros(num_features)

    def update(self, phi, phi_prime, cumulant, gamma, rho, **kwargs):
        alpha = self.alpha.next()
        beta = self.beta.next()
        temp = self.theta

        self.delta = (cumulant + gamma * np.dot(phi_prime, self.theta) -
                      np.dot(phi, self.theta))
        self.e = rho * (self.old_gamma * self.lmbda * self.e
                        + alpha * (1 - rho * self.old_gamma *
                                   self.lmbda *
                                   np.dot(phi, self.e))
                        * phi)
        self.e_grad = rho * (self.lmbda * self.old_gamma * self.e_grad + phi)
        self.e_w = ((self.old_rho * self.old_gamma *
                     self.lmbda * self.e_w)
                    + beta * (1 - self.old_rho *
                              self.old_gamma * self.lmbda *
                              np.dot(phi, self.e_w) * phi))
        self.tderr_elig = self.delta * self.e

        self.theta += (self.tderr_elig +
                       (self.e - alpha * rho * phi) *
                       np.dot(self.theta - self.old_theta,
                              phi) -
                       alpha * gamma * (1 - self.lmbda) *
                       np.dot(self.w, self.e_grad) *
                       phi_prime)
        self.w += (rho * self.delta * self.e_w -
                   beta * np.dot(phi, self.w) * phi)

        self.old_gamma = gamma
        self.old_rho = rho
        self.old_theta = temp

        # for compatibility with calculating RUPEE for control gvfs
        return phi

    def predict(self, phi):
        return np.dot(phi, self.theta)
