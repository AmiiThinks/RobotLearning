import random
import tiles3
import itertools
import numpy as np
import rospy 
import visualize_pixels
import cv2

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
    DIFF_BW_RGB = 256/NUM_TILINGS

    # constants relating to image size recieved
    IMAGE_LI = 480 # lines
    IMAGE_CO = 640 # columns


class StateManager(object):
    def __init__(self):
        self.ihts = [tiles3.IHT(StateConstants.NUM_INTERVALS) for i in
                     xrange(StateConstants.NUM_RANDOM_POINTS * 3)]

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
    def get_state_representation(self, image, bumper_information, action):
        phi = np.zeros(StateConstants.TOTAL_FEATURE_LENGTH, dtype=np.bool)

        # setting the bias unit
        phi[StateConstants.BIAS_FEATURE_INDEX] = True 

        # adding image data to state
        if image is None or len(image) == 0 or len(image[0]) == 0:
            rospy.loginfo("empty image has no representation")
            if self.last_image_raw is None:
                return phi
            else:
                return self.get_state_representation(self.last_image_raw,
                                                     bumper_information,
                                                     action)

        self.last_image_raw = image

        rgbpoints_raw = image[self.pixel_mask].flatten()

        for color_index in xrange(len(rgbpoints_raw)):
            rbg = [rgbpoints_raw[color_index] / StateConstants.DIFF_BW_RGB]
            tiles = tiles3.tiles(self.ihts[color_index], 
                                 StateConstants.NUM_TILINGS,
                                 rbg)

            for i in xrange(len(tiles)):
                phi[color_index * StateConstants.NUM_FEATURES_PER_COL_VAL + i * StateConstants.NUM_TILINGS + tiles[i]] = True

        rospy.loginfo("PHI SENT")
        rospy.loginfo(phi)

        return phi

    def get_num_tilings(self):
        return StateConstants.NUM_TILINGS

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
        sr = state_rep.get_state_representation(image, 1)
