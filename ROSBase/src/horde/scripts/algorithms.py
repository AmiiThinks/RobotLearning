import numpy as np
import random
from policy import Policy
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import Twist, Vector3
import sys

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
        self.delta = 0
        self.action_space = []

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
        self.delta = delta

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
        self.delta = 0

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
        self.delta = delta

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

class eGreedy(Policy):
    def __init__(self, epsilon = 0, ):
        Policy.__init__(self)
        self.epsilon = epsilon

    def take_action(self, phi, learned_policy, action_space, theta):
        # select a random number between 0 and 1
        random_number = random.uniform(0, 1)
        if random_number < self.epsilon:
            random_action = action_space[random.randint(0,len(action_space)-1)]
            return random_action, self.epsilon/len(action_space)

        greedy_action,_ = learned_policy.take_action(phi, learned_policy, action_space, theta)
        # take the action here
        return greedy_action, 1 - self.epsilon + self.epsilon/len(action_space)

class Learned_Policy(Policy):
    def __init__(self):
        Policy.__init__(self)

    def take_action(self, phi, learned_policy, action_space, theta):
        self.action_space = action_space
        greedy_action = action_space[0]
        if phi != None:
            return action_space[0], -1

        for action in action_space:
            if np.dot(theta, self.get_representation(phi,action)) >= np.dot(self.theta, self.get_representation(phi,action)):
                greedy_action = action
        return greedy_action, -1

    def get_representation(self, state, action):
        representation = []
        state = np.ndarray.tolist(state)
        for index, current_action in enumerate(self.action_space):
            if current_action == action:
                representation = representation + state
            else:
                representation = representation + [0]*len(state)
        return np.asarray(representation)

class GreedyGQ:
    """ From Maei/Sutton 2010, with additional info from Adam White. """

    def __init__(self,theta,gamma,_lambda,cumulant,alpha,beta,epsilon):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        self.epsilon = epsilon
        self.behavior_policy = eGreedy(epsilon = self.epsilon)
        self.learned_policy = Learned_Policy()
        self.theta = theta
        self.gamma = gamma
        self._lambda = _lambda
        self.learning_rate = alpha
        self.secondary_learning_rate = beta
        self.cumulant = cumulant
        self.td_error = 0
        self.sec_weights = np.zeros(14403*5)
        self.etrace = np.zeros(14403*5)
        # remove multiple phi's, we're using phi, rest of the code needs, self.phi
        self._phi = np.zeros(14403)
        self.phi = np.zeros(14403*5)

        self.action_space = [Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                        Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0)), # forward
                        Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0)), # backward
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, 1.5)), # turn acw/cw
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, -1.5)) # turn cw/acw
                        ]

    def take_action(self, phi_prime):
        action, mu = self.behavior_policy.take_action(phi_prime, self.learned_policy,self.action_space, self.theta)
        return action, mu

    def take_random_action(self):
        random_action = self.action_space[random.randint(0,len(self.action_space)-1)]
        return random_action, 1/len(self.action_space)

    def update(self, state, action, observation, next_state):
        reward = self.cumulant(observation)
        gamma = self.gamma(observation)
        learned_policy = self.learned_policy
        behavior_policy = self.behavior_policy
        self.phi = self.get_representation(state,action)

        if reward == 1:
            print 'Episode finished'
            self.etrace = np.zeros(14403*5)
            # give the signal to stop
            return True

        # A_{t+1} update
        next_greedy_action = action
        for action in self.action_space:
            if np.dot(self.theta, self.get_representation(next_state,action)) >= np.dot(self.theta, self.get_representation(next_state,next_greedy_action)):
                next_greedy_action = action

        # phi_bar update
        phi_bar = self.get_representation(next_state,next_greedy_action)

        # delta_t update
        self.td_error = reward + gamma * np.dot(self.theta, phi_bar) - np.dot(self.theta,self.phi)
        
        previous_greedy_action = action
        for temp_action in self.action_space:
            if np.dot(self.theta, self.get_representation(state,temp_action)) >= np.dot(self.theta, self.get_representation(state,previous_greedy_action)):
                previous_greedy_action = temp_action

        # rho_t (responsibility) update
        if action == previous_greedy_action:
            responsibility = 1/(1-self.epsilon+self.epsilon/len(self.action_space))
        else:
            responsibility = 0

        # e_t update
        self.etrace *= gamma * self._lambda * responsibility
        self.etrace += self.phi #(phi_t) 
                
        # theta_t update
        self.theta += self.learning_rate * (self.td_error * self.etrace - 
                        gamma * (1 - self._lambda) * np.dot(self.sec_weights, self.phi) * phi_bar)
        
        # w_t update
        self.sec_weights += self.secondary_learning_rate * (self.td_error * self.etrace - np.dot(self.sec_weights, self.phi) * self.phi)
        return False

    def get_representation(self, state, action):
        representation = []
        state = np.ndarray.tolist(state)
        for index, current_action in enumerate(self.action_space):
            if current_action == action:
                representation = representation + state
            else:
                representation = representation + [0]*len(state)
        return np.asarray(representation)

