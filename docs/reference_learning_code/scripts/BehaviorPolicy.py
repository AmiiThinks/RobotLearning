"""
Note: We will have to do something like this with our behavior policy.
"""

from random import randint

from horde.msg import StateRepresentation

class BehaviorPolicy:
    """
    0 = stay,
    1 = move right
    2 = move left
    """
    def __init__(self):
        self.lastAction = 0
        self.i = 0

    def policy(self, state):
        #return self.randomPolicy(state)
        return self.backAndForthPolicy(state)
        #return self.fiveRightPolicy(state)
        #return self.slowBackAndForthPolicy(state)

    def randomPolicy(self, state):
        if (randint(0,9) < 3):
            #50 of the time, do the previous action
            return self.lastAction
        else:
            #else do a random - left or right - action
            #TODO - randint including 0 - the rest position.
            self.lastAction = randint(1,2)
            return self.lastAction

    def backAndForthPolicy(self, state):
        if self.lastAction == 1:
            self.lastAction = 2
            return 2
        else:
            self.lastAction = 1
            return 1

    def fiveRightPolicy(self, state):
        self.i = self.i + 1
        if self.i % 5 ==0:
            #switch action
            if self.lastAction == 1:
                self.lastAction = 2
            else:
                self.lastAction = 1
            i = 0

        return self.lastAction

