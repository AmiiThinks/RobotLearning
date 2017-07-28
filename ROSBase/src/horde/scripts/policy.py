from __future__ import division
import numpy as np
import tools

from state_representation import StateConstants

class Policy:
    def __init__(self, 
                 action_space, 
                 feature_indices,
                 value_function=None,
                 action_equality=tools.equal_twists,
                 *args,
                 **kwargs):

        self.action_space = np.asarray(action_space)
        self.pi = np.ones(action_space.size) / action_space.size
        self.value = value_function
        self.equals = action_equality
        self.feature_indices = feature_indices
        self.last_index = 0

    def update(self, phi, observation, *args, **kwargs):
        """ Overwrite this function """
        # update the probability of chosing each action
        if self.value is not None:
            phi = phi[self.feature_indices]

            q_fun = np.vectorize(lambda a: self.value(phi, a))
            q_values = q_fun(self.action_space)

            self.pi = np.array(q_values) / sum(q_values)

    def get_probability(self, action, *args, **kwargs):
        # return the probability of taking a given action
        equal_action = lambda i: self.equals(action, self.action_space[i])
        self.last_index=filter(equal_action, range(self.action_space.size))[0]
        return self.pi[self.last_index]

    def choose_action(self, *args, **kwargs):
        return np.random.choice(self.action_space, p=self.pi)
