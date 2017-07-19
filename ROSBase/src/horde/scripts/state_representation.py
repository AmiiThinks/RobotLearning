import math
import numpy as np
import random
import rospy 

import tiles3
from tools import timing

# np.set_printoptions(threshold=np.nan)

"""
# StateManager:

Picks NUM_RANDOM_POINTS random rgb values from an image and tiles those
values to obtain the state representation
"""

class StateConstants:

    # image tiles
    NUM_RANDOM_POINTS = 300
    CHANNELS = 3
    NUM_IMAGE_TILINGS = 4
    NUM_IMAGE_INTERVALS = 4 
    SCALE_RGB = NUM_IMAGE_TILINGS/256.0
    IMAGE_IHT_SIZE = (NUM_IMAGE_INTERVALS + 1) * NUM_IMAGE_TILINGS
    TOTAL_PIXEL_FEATURE_LENGTH = NUM_RANDOM_POINTS * 3 * IMAGE_IHT_SIZE
    IMAGE_START_INDEX = 0

    # constants relating to image size recieved
    IMAGE_LI = 480 # rows
    IMAGE_CO = 640 # columns

    # IMU tiles
    NUM_IMU_TILINGS = 8
    NUM_IMU_TILES = 40
    SCALE_IMU = NUM_IMU_TILES/2.0 # range is [-1, 1]
    IMU_IHT_SIZE = (NUM_IMU_TILES + 1) * NUM_IMU_TILINGS
    IMU_START_INDEX = IMAGE_START_INDEX + TOTAL_PIXEL_FEATURE_LENGTH

    # Odom tiles
    NUM_ODOM_TILINGS = 8
    NUM_ODOM_TILES = 5
    SCALE_ODOM = NUM_ODOM_TILES/2.0 # range is [0, 2]
    ODOM_IHT_SIZE = (NUM_ODOM_TILES + 1) * (NUM_ODOM_TILES + 1) * NUM_ODOM_TILINGS
    ODOM_START_INDEX = IMU_START_INDEX + IMU_IHT_SIZE

    # the 1 represents the bias unit, 3 for bump
    TOTAL_FEATURE_LENGTH = TOTAL_PIXEL_FEATURE_LENGTH + IMU_IHT_SIZE + ODOM_IHT_SIZE + 3 + 1


class StateManager(object):
    def __init__(self):

        num_img_ihts = StateConstants.NUM_RANDOM_POINTS * StateConstants.CHANNELS
        img_iht_size = StateConstants.IMAGE_IHT_SIZE
        self.img_ihts = [tiles3.IHT(img_iht_size) for _ in xrange(num_img_ihts)]

        self.imu_iht = tiles3.IHT(StateConstants.IMU_IHT_SIZE)
        self.odom_iht = tiles3.IHT(StateConstants.ODOM_IHT_SIZE)


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
        self.last_imu_raw = None
        self.last_odom_raw = None

    @timing
    def get_phi(self, image, bump, ir, imu, odom, weights = None):

        phi = np.zeros(StateConstants.TOTAL_FEATURE_LENGTH)

        # IMAGE STUFF
        # check if there is an image
        no_image = image is None or len(image) == 0 or len(image[0]) == 0

        # adding image data to state
        if no_image:
            rospy.logwarn("Image is empty.")
            if self.last_image_raw is not None:
                image = self.last_image_raw

        if image is not None:
            self.last_image_raw = image 
            rgb_points = image[self.pixel_mask].flatten().astype(float)
            rgb_points *= StateConstants.SCALE_RGB
            rgb_inds = np.arange(StateConstants.NUM_RANDOM_POINTS * 3)

            tile_inds = [tiles3.tiles(self.img_ihts[i], 
                                      StateConstants.NUM_IMAGE_TILINGS, 
                                      [rgb_points[i]]) for i in rgb_inds]

            # tile_inds = np.ones((900,4), dtype=int)

            rgb_inds *= StateConstants.IMAGE_IHT_SIZE
            # tiling_inds = np.arange(start=0, 
            #                         stop=math.pow(StateConstants.NUM_IMAGE_TILINGS, 2), 
            #                         step=StateConstants.NUM_IMAGE_TILINGS,
            #                         dtype=int).reshape(1,-1)

            indices = tile_inds + rgb_inds[:, np.newaxis] # + tiling_inds
            phi[indices.flatten()] = True


        # IMU STUFF
        if imu is None:
            rospy.logwarn("No imu value.")
            if self.last_imu_raw is not None:
                imu = self.last_imu_raw

        if imu is not None:
            self.last_imu_raw = imu
            indices = np.array(tiles3.tiles(self.imu_iht, 
                                            StateConstants.NUM_IMU_TILINGS, 
                                            [imu*StateConstants.SCALE_IMU]))

            phi[indices + StateConstants.IMU_START_INDEX] = True

        # ODOM STUFF
        if odom is None:
            rospy.logwarn("No odom value.")
            if self.last_odom_raw is not None:
                odom = self.last_odom_raw

        if odom is not None:
            self.last_odom_raw = odom
            indices = np.array(tiles3.tiles(self.odom_iht,
                                            StateConstants.NUM_ODOM_TILINGS,
                                            odom * StateConstants.SCALE_ODOM))

            phi[indices + StateConstants.ODOM_START_INDEX] = True

        # bump
        if bump is not None:
            phi[-4:-1] = bump

        # bias unit
        phi[-1] = True
        
        return phi

    def get_observations(self, bump, ir, **kwargs):
        observations = {'bump': bump if bump else (0,0,0),
                          'ir': ir if ir else (0,0,0),
                       }

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
