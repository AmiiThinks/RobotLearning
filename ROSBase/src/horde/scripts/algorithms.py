import numpy as np

class GTD:

    """
    Represents a true online temporal difference lambda learning agent.
    From the summer 2016 Robot Knowledge Project. 

    See testTime.py for usage.
    """

    def __init__(self, alpha, beta, lambda_, gamma, theta, phi):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.old_gamma = np.zeros(self.gamma.shape)
        self.lambda_ = np.array(lambda_)
        self.old_lambda = np.zeros(self.lambda_.shape)
        self.theta = np.atleast_2d(theta)
        self._phi = np.array(np.copy(phi)) # use a copy of phi
        self._e = np.zeros(np.shape(self.theta))
        self._w = np.zeros(np.shape(self.theta))

    def update(self, phi_prime, reward, rho, alpha=None, beta=None, lambda_=None, gamma=None):
        """
        Updates the parameter vector for a new observation. If any optional
        values are set then the new value of the optional is used for this and
        future calls that do not set the same optional value.
        """
        # set optional values
        if alpha is not None:
            self.alpha = np.array(alpha)
        if beta is not None:
            self.beta = np.array(beta)
        if lambda_ is not None:
            self.lambda_ = np.array(lambda_)
        if gamma is not None:
            self.gamma = np.array(gamma)
        rho = np.array(rho)
        # calculate V and V_prime
        V = np.dot(self.theta, self._phi)
        V_prime = np.dot(self.theta, phi_prime)

        # calculate delta
        delta = reward + self.gamma * V_prime - V

        # update eligibility traces
        self._e *= (rho * self.old_lambda * self.old_gamma)[..., np.newaxis]
        self._e += np.outer(rho, self._phi)

        # update theta
        self.theta += (self.alpha*delta*self._e.T).T
        self.theta -= np.outer(self.alpha*self.gamma*(1-self.lambda_)*np.sum(self._e*self._w, axis = 1), phi_prime)

        #update w
        self._w -= np.outer(self.beta*np.dot(self._w, self._phi), self._phi)
        self._w += (self.beta*delta*self._e.T).T

        # update values
        self._phi = np.array(np.copy(phi_prime))
        self.old_gamma = np.array(np.copy(self.gamma))
        self.old_lambda = np.array(np.copy(self.lambda_))

    def predict(self, phi):
        """
        Returns the current prediction for a given set of features phi.
        """
        return np.dot(self.theta, phi)

class BinaryGTD:

    """
    Represents a true online temporal difference lambda learning agent.
    From the summer 2016 Robot Knowledge Project. 

    See testTime.py for usage.
    """

    def __init__(self, alpha, beta, lambda_, gamma, theta, phi):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.old_gamma = np.zeros(self.gamma.shape)
        self.lambda_ = np.array(lambda_)
        self.old_lambda = np.zeros(self.lambda_.shape)
        self.theta = np.atleast_2d(theta)
        self._phi = np.array(np.copy(phi)) # use a copy of phi
        self._e = np.zeros(np.shape(self.theta))
        self._w = np.zeros(np.shape(self.theta))

    def update(self, phi_prime, reward, rho, alpha=None, beta=None, lambda_=None, gamma=None):
        """
        Updates the parameter vector for a new observation. If any optional
        values are set then the new value of the optional is used for this and
        future calls that do not set the same optional value.
        """
        # set optional values
        if alpha is not None:
            self.alpha = np.array(alpha)
        if beta is not None:
            self.beta = np.array(beta)
        if lambda_ is not None:
            self.lambda_ = np.array(lambda_)
        if gamma is not None:
            self.gamma = np.array(gamma)
        rho = np.array(rho)
        # calculate V and V_prime
        V = np.sum(self.theta[:,self._phi], axis = 1)
        V_prime = np.sum(self.theta[:,phi_prime], axis = 1)

        # calculate delta
        delta = reward + self.gamma * V_prime - V

        # update eligibility traces
        self._e *= (rho * self.old_lambda * self.old_gamma)[..., np.newaxis]
        self._e[:, self._phi] += rho[..., np.newaxis]

        # update theta
        self.theta += (self.alpha*delta*self._e.T).T
        self.theta[:, phi_prime] -=(self.alpha*self.gamma*(1-self.lambda_)*np.sum(self._e*self._w, axis = 1))[..., np.newaxis]

        #update w
        self._w[:, self._phi] -= (self.beta*np.sum(self._w[:, self._phi], axis = 1))[..., np.newaxis]
        self._w += (self.beta*delta*self._e.T).T

        # update values
        self._phi = np.array(np.copy(phi_prime))
        self.old_gamma = np.array(np.copy(self.gamma))
        self.old_lambda = np.array(np.copy(self.lambda_))

    def predict(self, phi):
        """
        Returns the current prediction for a given set of features phi.
        """
        return np.sum(self.theta[:,phi], axis = 1)