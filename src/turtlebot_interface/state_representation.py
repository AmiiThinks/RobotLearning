import random
import tiles3
import itertools
import numpy as np

from functools import wraps
from time import time

np.set_printoptions(threshold=np.nan)

# DEBUG: This is for debugging purposes
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print 'func:%r args:[%r, %r] took: %2.4f sec' % \
        # (f.__name__, args, kw, te-ts)
        print 'func:%r took: %2.4f sec' % (f.__name__, te-ts)
        return result
    return wrap

"""
# StateRepresentation.py:

Picks NUM_RANDOM_POINTS random rgb values from an image and tiles those
values to obtain the state representation
"""

NUM_RANDOM_POINTS = 300
NUM_TILINGS = 4
NUM_INTERVALS = 4 
NUM_FEATURES_PER_COL_VAL = NUM_TILINGS * NUM_INTERVALS

# regards the generalization between tile dimensions
DIFF_BW_R = 100
DIFF_BW_G = 100
DIFF_BW_B = 100
DIFF_BW_RGB = 256/NUM_TILINGS
DIFF_BW_BUMP = 1


class StateRepresentation:
    def __init__(self):
        self.ihts = [tiles3.IHT(NUM_INTERVALS) for i in
                     xrange(NUM_RANDOM_POINTS * 3)]

    # Grabs a number of random pixels from an image (see NUM_RANDOM_POINTS)
    # Not used by user
    # image: a 2D array with
    @timing
    def random_points(self, image):
        
        random_points = []

        for p in range(NUM_RANDOM_POINTS):
            p1 = random.randint(0, len(image) - 1)
            p2 = random.randint(0, len(image[0]) - 1)

            random_points.append(image[p1][p2])

        return random_points

    # Gets the state representation of NUM_RANDOM_POINTS pixels
    @timing
    def get_state_representation(self, image, action):
        if image is None or len(image) == 0 or len(image[0]) == 0:
            print ("empty image has no representation")
            return []

        points = self.random_points(image)
        state_representation_raw = \
            np.zeros(NUM_RANDOM_POINTS * 3 * NUM_FEATURES_PER_COL_VAL)
        rgbpoints_raw = np.array(list(itertools.chain.from_iterable(points)))
        
        for color_index in xrange(len(rgbpoints_raw)):
            tiles = tiles3.tiles(self.ihts[color_index], NUM_TILINGS,
                                 [rgbpoints_raw[color_index] / DIFF_BW_RGB])
            index = 0 
            for t in tiles:
                state_representation_raw[color_index * NUM_FEATURES_PER_COL_VAL
                                         + index * NUM_INTERVALS + t] = 1
                index+=1
        return state_representation_raw

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
    state_rep = StateRepresentation()

    for i in range(10):
        image = DEBUG_generate_rand_image()
        sr = state_rep.get_state_representation(image, 1)
