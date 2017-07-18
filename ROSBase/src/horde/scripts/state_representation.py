import random
import tiles3
import numpy as np
import rospy 

from tools import timing

# np.set_printoptions(threshold=np.nan)

"""
# StateManager:

Picks NUM_RANDOM_POINTS random rgb values from an image and tiles those
values to obtain the state representation
"""

class StateConstants:
    NUM_RANDOM_POINTS = 300
    NUM_TILINGS = 4
    NUM_INTERVALS = 4 
    NUM_FEATURES_PER_COL_VAL = NUM_TILINGS * NUM_INTERVALS

    TOTAL_PIXEL_FEATURE_LENGTH = NUM_RANDOM_POINTS * 3 * NUM_FEATURES_PER_COL_VAL
    BIAS_FEATURE_INDEX = TOTAL_PIXEL_FEATURE_LENGTH

    # the 3 represents the number quantity of values in rgb
    # the 1 represents the bias unit
    TOTAL_FEATURE_LENGTH = TOTAL_PIXEL_FEATURE_LENGTH + 1

    # regards the generalization between tile dimensions
    DIFF_BW_RGB = NUM_TILINGS/256.0

    # constants relating to image size recieved
    IMAGE_LI = 480 # lines
    IMAGE_CO = 640 # columns


class StateManager(object):
    def __init__(self):
        self.ihts = [tiles3.IHT(StateConstants.NUM_INTERVALS) for _ in xrange(StateConstants.NUM_RANDOM_POINTS * 3)]

        self.iht = tiles3.IHT(StateConstants.NUM_INTERVALS)

        # set up mask to chose pixels
        num_pixels = StateConstants.IMAGE_LI*StateConstants.IMAGE_CO
        num_chosen = StateConstants.NUM_RANDOM_POINTS
        self.chosen_indices = np.random.choice(a=num_pixels, 
                                          size=num_chosen, 
                                          replace=False)
        self.pixel_mask = np.zeros(num_pixels, dtype=np.bool)
        self.pixel_mask[self.chosen_indices] = True
        self.pixel_mask = self.pixel_mask.reshape(StateConstants.IMAGE_LI, 
                                                  StateConstants.IMAGE_CO)

        self.last_image_raw = None
        self.last_bumper_raw = None


    @timing
    def get_phi(self, image, weights = None):
        phi = np.zeros(StateConstants.TOTAL_FEATURE_LENGTH, dtype=np.bool)

        # setting the bias unit
        phi[StateConstants.BIAS_FEATURE_INDEX] = True 

        # check if there is an image
        no_image = image is None or len(image) == 0 or len(image[0]) == 0

        # adding image data to state
        if no_image:
            rospy.loginfo("empty image has no representation")
            if self.last_image_raw is None:
                return phi
        else:
            self.last_image_raw = image 

        rgb_points = image[self.pixel_mask].flatten()
        rgb_points *= StateConstants.DIFF_BW_RGB
        rgb_inds = np.arange(StateConstants.NUM_RANDOM_POINTS * 3)

        # tiles = lambda k: tiles3.tiles(self.iht, 
        #                                StateConstants.NUM_TILINGS, 
        #                                [k])
        # tile_inds = map(tiles, rgb_points)
        tile_inds = [tiles3.tiles(self.iht, 
                                       StateConstants.NUM_TILINGS, 
                                       [k]) for k in rgb_points]

        # tile_inds = np.ones((900,4), dtype=int)

        rgb_inds *= StateConstants.NUM_FEATURES_PER_COL_VAL
        tiling_inds = np.arange(0, StateConstants.NUM_TILINGS ** 2, step = StateConstants.NUM_TILINGS).reshape(1, -1)
        # tiling_inds *= StateConstants.NUM_TILINGS

        indices = tile_inds + rgb_inds[:, np.newaxis] + tiling_inds

        phi[indices.flatten()] = True

        return phi

    def get_observations(self, bumper_information):
        observations = dict()
        if bumper_information is None:
            if self.last_bumper_raw is None:
                bumper_information = (0,0,0)
            else:
                bumper_information = self.last_bumper_raw
        observations["bump"] = bumper_information
        return observations

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
        sr = state_rep.get_phi(image)
