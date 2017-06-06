import random
import tiles3
import itertools
import numpy as np
import rospy 

from tools import timing

# np.set_printoptions(threshold=np.nan)

"""
# StateManager:

Picks NUM_RANDOM_POINTS random rgb values from an image and tiles those
values to obtain the state representation
"""

NUM_RANDOM_POINTS = 300
NUM_TILINGS = 4
NUM_INTERVALS = 4 
NUM_FEATURES_PER_COL_VAL = NUM_TILINGS * NUM_INTERVALS
NUM_BUMPERS = 3
NUM_FEATURES_PER_BUMPER = 1
TOTAL_FEATURE_LENGTH = NUM_RANDOM_POINTS * 3 *  \
	NUM_FEATURES_PER_COL_VAL + NUM_BUMPERS * NUM_FEATURES_PER_BUMPER

# regards the generalization between tile dimensions
DIFF_BW_R = 100
DIFF_BW_G = 100
DIFF_BW_B = 100
DIFF_BW_RGB = 256/NUM_TILINGS
DIFF_BW_BUMP = 1


class StateManager:
    def __init__(self):
        self.ihts = [tiles3.IHT(NUM_INTERVALS) for i in
                     xrange(NUM_RANDOM_POINTS * 3)]

    # Grabs a number of random pixels from an image (see NUM_RANDOM_POINTS)
    # Not used by user
    # image: a 2D array with
    def random_points(self, image):
        
        random_points = []

        for p in range(NUM_RANDOM_POINTS):
            p1 = random.randint(0, len(image) - 1)
            p2 = random.randint(0, len(image[0]) - 1)

            random_points.append(image[p1][p2])

        return random_points

    # Gets the state representation of NUM_RANDOM_POINTS pixels
    @timing
    def get_state_representation(self, image, lbump, cbump, rbump, action):
        if image is None or len(image) == 0 or len(image[0]) == 0:
            rospy.loginfo("empty image has no representation")
            return []

        points = self.random_points(image)
        state_representation_raw = \
            np.zeros(TOTAL_FEATURE_LENGTH)
        rgbpoints_raw = np.array(list(itertools.chain.from_iterable(points)))
        
        # adding bumper data to the state
        state_representation_raw[0] = lbump
        state_representation_raw[1] = cbump
        state_representation_raw[2] = rbump

        for color_index in xrange(len(rgbpoints_raw)):
            tiles = tiles3.tiles(self.ihts[color_index], NUM_TILINGS,
                                 [rgbpoints_raw[color_index] / DIFF_BW_RGB])
            index = 0 
            for t in tiles:
                state_representation_raw[color_index * NUM_FEATURES_PER_COL_VAL
                                         + index * NUM_INTERVALS + t + 3] = 1
                index+=1

        return state_representation_raw

    def get_num_tilings(self):
        return NUM_TILINGS

# This is a debugging function. It just generates a random image.
@timing
def DEBUG_generate_rand_image():
    dimensions = [1080, 1080]

    return_matrix = [[0 for i in xrange(dimensions[1])] for j in
                     xrange(dimensions[0])]

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            color = [random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255)]
            return_matrix[i][j] = color

    return return_matrix

if __name__ == "__main__":
    state_rep = StateManager()

    for i in range(10):
        image = DEBUG_generate_rand_image()
        sr = state_rep.get_state_representation(image, 1)
