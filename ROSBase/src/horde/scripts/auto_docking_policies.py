import multiprocessing as mp
import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3


class eGreedy():
    def __init__(self, theta, learned_policy, epsilon = 0,action_space=[]):
        self.epsilon = epsilon
        self.theta = theta
        self.action_space = action_space
        self.learned_policy = learned_policy

    def __call__(self, phi, *args ,**kwargs):
        # select a random number between 0 and 1
        random_number = random.uniform(0, 1)
        if random_number < self.epsilon:
            print 'taking random action in behaviour policy'
            random_action = self.action_space[random.randint(0,len(self.action_space)-1)]
            return random_action, self.epsilon/len(self.action_space)

        greedy_action,_ = self.learned_policy(phi, self.action_space, self.theta)
        print 'taking greedy action in behaviour policy, action is : ', self.action_space.index(greedy_action)
        # take the action here
        return greedy_action, 1 - self.epsilon + self.epsilon/len(self.action_space)

class Learned_Policy():
    def __init__(self):
        pass

    def __call__(self, phi, action_space, theta):
        self.action_space = action_space
        greedy_action = action_space[1]
        if phi == None:
            print 'Warning: phi is None'
            return action_space[0], -1

        equal_actions = []
        if np.count_nonzero(theta) == 0:
            print 'Theta is zero'
        for action in action_space:
            print len(theta), len(self.get_representation(phi,action))
            print 'action_value: ', np.dot(theta, self.get_representation(phi,action)), 'greedy_action value: ' ,np.dot(theta, self.get_representation(phi,greedy_action))
            if np.dot(theta, self.get_representation(phi,action)) == np.dot(theta, self.get_representation(phi,greedy_action)):
                equal_actions.append(action)
                print 'equal'
            if np.dot(theta, self.get_representation(phi,action)) > np.dot(theta, self.get_representation(phi,greedy_action)):
                greedy_action = action
                equal_actions = [action]
        greedy_action = self.action_space[random.randint(0,len(equal_actions)-1)]
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
