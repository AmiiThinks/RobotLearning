"""Module containing parent class for policies.
"""

from __future__ import division

import numpy as np

import tools

class Policy:
    """Parent class for policies.

    Inherit this class to make a policy. Action selection is based on
    maintaining a ``pi`` array which holds action selection
    probabilities.

    Attributes:
        action_space (numpy array of action): Numpy array containing
            all actions available to any agent.
        value_function (optional): A function used by the Policy to 
            update values of pi. This is usually a value function 
            learned by a GVF.
        action_equality (optional): The function used to compare two 
            action objects to determine whether they are equivalent. 
            Returns True if the actions are equivalent and False 
            otherwise.
        feature_indices (numpy array of bool, optional): The indices 
            of the feature vector corresponding to the indices used by 
            the ``value_function``.
        pi (numpy array of float): Numpy array containing probabilities
            corresponding to the actions at the corresponding index in
            ``action_space``. Not passed to init.
        last_index (int): The index of the last action chosen by the
            policy. Not passed to init.
    """
    def __init__(self, 
                 action_space, 
                 feature_indices=None,
                 value_function=None,
                 action_equality=tools.equal_twists,
                 *args,
                 **kwargs):

        self.action_space = np.asarray(action_space)
        self.pi = np.ones(action_space.size) / action_space.size
        self.value_function = value_function
        self.action_equality = action_equality
        self.feature_indices = feature_indices
        self.last_index = 0

    def update(self, phi, observation, *args, **kwargs):
        """Updates the probilities of taking each action

        This function should be replaced when creating a new policy. It
        takes a state ``(phi, observation)`` and modifies the ``pi``
        array accordingly. 

        Args:
            phi (numpy array of bool): Binary feature vector.
            observation (dictionary): User-defined dictionary containing
                miscellaneous information about the state that should
                not be included in the feature vector ``phi``.
            *args: Ignored.
            **kwargs: Ignored. 
        """
        if self.value_function is not None:
            phi = phi[self.feature_indices]

            q_fun = np.vectorize(lambda a: self.value_function(phi, a))
            q_values = q_fun(self.action_space)

            self.pi = np.array(q_values) / q_values.sum()

    def get_probability(self, action, choice=True, *args, **kwargs):
        """Get the probability of taking the provided action.

        This function can usually be used without being overwritten.
        Throws an error if the provided action is not equal to an action
        in ``action_space`` according to ``action_equality``.

        Args:
            action (action): Find the probability of this action.
            choice (bool): If set to true, updates ``last_index``.
            *args: Ignored.
            **kwargs: Ignored. 

        Returns:
            Float from ``pi`` corresponding to ``action``. 
        """
        equal_action = lambda i: self.action_equality(action, 
                                                      self.action_space[i])
        indices  = list(filter(equal_action, range(self.action_space.size)))
        assert len(indices) > 0

        index = indices[0]

        if choice:
            self.last_index = index

        return self.pi[index]

    def choose_action(self, *args, **kwargs):
        """Updates ``last_index`` and chooses an action according to ``pi``.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Action at the sampled index.
        """
        self.last_index = np.random.choice(self.action_space.size, p=self.pi)
        return self.action_space[self.last_index]
