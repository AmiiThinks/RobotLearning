"""
Updated Behaviour Policy by Michele
"""

from horde.msg import StateRepresentation

class BehaviorPolicy:
    """
    0 = stay
    1 = go forward
    2 = turn (right)
    """
    def __init__(self):
        self.lastAction = 0
        self.i = 0

    def policy(self, state):
        return self.forwardPolicy(state)
        #return turnRightPolicy(state)
    
    def forwardPolicy(self, state):
        self.lastAction = 1
        return 1

    def turnRightPolicy(self, state):
        self.lastAction = 2
        return 2
