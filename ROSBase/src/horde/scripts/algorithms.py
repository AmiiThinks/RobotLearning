import numpy as np
import random
from policy import Policy
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import Twist, Vector3

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

    def take_action(self, phi, learned_policy):
        # select a random number between 0 and 1
        random_number = random.uniform(0, 1)
        if random_number < epsilon:
            random_action = action_space[randint(0,len(action_space)-1)]
            return random_action, self.epsilon/len(action_space)

        greedy_action,_ = learned_policy.take_action()
        # take the action here
        return greedy_action, 1-self.epsilon+self.epsilon/len(action_space)

class Learned_Policy(Policy):
    def __init__(self):
        Policy.__init__(self)

    def take_action(self, phi, learned_policy):
        greedy_action = action_space[0]
        for action in action_space:
            if dot(self._theta, self.get_representation(state,action)) >= dot(self._theta, self.get_representation(state,greedy_action)):
                greedy_action = action

        return greedy_action, _

class GreedyGQ:
    """ From Maei/Sutton 2010, with additional info from Adam White. """

    def __init__(self,_theta,rewardDiscount,_lambda,cumulant,learningRate,epsilon):
        """
        Constructs a new agent with the given parameters. Note that a copy of
        phi is created during the construction process.
        """
        self.epsilon = epsilon
        self._behaviorPolicy = eGreedy(epsilon = self.epsilon)
        self._greedyPolicy = Learned_Policy()
        self._theta = _theta
        self._rewardDiscount = rewardDiscount
        self._lambda = None
        self._learningRate = None
        self._sec_weights = None
        self._td_error = None
        self._etraces = None
        self.cumulant = cumulant
        action_space = [Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)),
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)),
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)),
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)),
                        Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
                        ]   
        
    def update(self, state, action, observation, next_state):
        reward = self.cumulant(observation)
        learned_policy = self._greedyPolicy
        behavior_policy = self._behaviorPolicy

        # A_t+1 update
        # I'm assuming self.get_representation(next_state,action) gives the state-action representation of that pair
        next_greedy_action = action
        for action in action_space:
            if dot(self._theta, self.get_representation(next_state,action)) >= dot(self._theta, self.get_representation(next_state,next_greedy_action)):
                next_greedy_action = action

        # phi_bar update
        phi_bar = self.get_representation(next_state,next_greedy_action)

        # delta_t update
        self._td_error = reward + self._rewardDiscount * rv_dot(self._theta, phi_bar) - dot(self._theta,self.get_representation(state,action))
        
        previous_greedy_action = action
        for action in action_space:
            if dot(self._theta, self.get_representation(state,action)) >= dot(self._theta, self.get_representation(state,previous_greedy_action)):
                previous_greedy_action = action

        # rho_t (responsibility) update
        if action == greedy_action:
            responsibility = 1/(1-self.epsilon+self.epsilon/len(action_space))
        else:
            responsibility = 0

        # e_t update
        self._etraces *= self._rewardDiscount * self._lambda * responsibility
        self._etraces += state #(phi_t) 
                
        # theta_t update
        self._theta += self._learningRate * (self._td_error * self._etraces - 
                        self._rewardDiscount * (1 - self._lambda) * rv_dot(self._sec_weights, self.get_representation(state,action)) * phi_bar)
        
        # w_t update
        self._sec_weights += self._learningRate * (td_error * self._etraces - rv_dot(self._sec_weights, phi) * phi)

    def get_representation(state, action):
        representation = []
        for current_action in self.action_space:
            if current_action == action:
                representation.append(state)
            else:
                representation.append(np.zeros(len(state)))
        return np.asarray(representation)

