import math
import numpy as np
import random
import rospy
from scipy.misc import comb

from CTiles import tiles
from tools import timing, get_next_pow2

# np.set_printoptions(threshold=np.nan)

"""
# StateManager:

Picks NUM_RANDOM_POINTS random rgb values from an image and tiles those
values to obtain the state representation
"""

class StateConstants:

    # image tiles
    NUM_RANDOM_POINTS = 100
    CHANNELS = 3
    NUM_IMAGE_TILINGS = 4
    NUM_IMAGE_INTERVALS = 4
    SCALE_RGB = NUM_IMAGE_TILINGS/256.0
    IMAGE_IHT_SIZE = get_next_pow2((NUM_IMAGE_INTERVALS + 1) * NUM_IMAGE_TILINGS)
    PIXEL_FEATURE_LENGTH = CHANNELS * IMAGE_IHT_SIZE
    TOTAL_IMAGE_FEATURE_LENGTH = NUM_RANDOM_POINTS * PIXEL_FEATURE_LENGTH
    IMAGE_START_INDEX = 0

    # constants relating to image size recieved
    IMAGE_LI = 480 # rows
    IMAGE_CO = 640 # columns

    # IMU tiles
    NUM_IMU_TILINGS = 1
    NUM_IMU_TILES = 6
    SCALE_IMU = (NUM_IMU_TILES)/2.0 # range is [-1, 1]
    IMU_IHT_SIZE = get_next_pow2((NUM_IMU_TILES + 1) * NUM_IMU_TILINGS)
    IMU_START_INDEX = IMAGE_START_INDEX + TOTAL_IMAGE_FEATURE_LENGTH

    # Odom tiles
    NUM_ODOM_TILINGS = 1
    NUM_ODOM_TILES = 5
    SCALE_ODOM = NUM_ODOM_TILES/2.0 # range is [0, 2]
    ODOM_IHT_SIZE = get_next_pow2((NUM_ODOM_TILES + 1) * (NUM_ODOM_TILES + 1) * NUM_ODOM_TILINGS)
    ODOM_START_INDEX = IMU_START_INDEX + IMU_IHT_SIZE

    # IR tiles
    IR_START_INDEX = ODOM_START_INDEX + ODOM_IHT_SIZE
    # IR_ITH_SIZE = 64*3
    IR_ITH_SIZE = 6*3

    # pixel pairs
    NUM_PP = comb(NUM_RANDOM_POINTS, 2, exact=True)
    NUM_PP_TILINGS = 4
    NUM_PP_INTERVALS = 4 
    SCALE_PP = NUM_PP_TILINGS/2 # [-1, 1]
    PP_IHT_SIZE = get_next_pow2((NUM_PP_INTERVALS + 1) * NUM_PP_TILINGS)
    PP_FEATURE_LENGTH = NUM_RANDOM_POINTS * PP_IHT_SIZE
    PP_START_INDEX = IR_START_INDEX + IR_ITH_SIZE

    # the 1 represents the bias unit, 3 for bump
    TOTAL_FEATURE_LENGTH = TOTAL_IMAGE_FEATURE_LENGTH + IMU_IHT_SIZE + ODOM_IHT_SIZE + IR_ITH_SIZE + 3 + 1 + PP_IHT_SIZE

    indices_in_phi = {'image':np.arange(0,TOTAL_IMAGE_FEATURE_LENGTH),
                        'imu':np.arange(IMU_START_INDEX,IMU_START_INDEX + IMU_IHT_SIZE),
                        'odom':np.arange(ODOM_START_INDEX,ODOM_START_INDEX + ODOM_IHT_SIZE),
                        'ir':np.arange(IR_START_INDEX,IR_START_INDEX+IR_ITH_SIZE),
                        'pixel_pairs': np.arange(PP_START_INDEX, PP_START_INDEX + PP_IHT_SIZE),
                        'bump':np.arange(IR_START_INDEX+IR_ITH_SIZE, IR_START_INDEX+IR_ITH_SIZE+3),
                        'bias':np.array([TOTAL_FEATURE_LENGTH-1]),
                        'last_action': np.array([], dtype=int)}

    num_active_features = {'image':NUM_RANDOM_POINTS*CHANNELS*NUM_IMAGE_TILINGS,
                        'imu':NUM_IMU_TILINGS,
                        'odom':NUM_ODOM_TILINGS,
                        'ir':6,
                        'bump':3,
                        'bias':1,
                        'last_action': 1,
                        'pixel_pairs': NUM_PP*NUM_PP_TILINGS}


class StateManager(object):
    def __init__(self, features_to_use):

        num_img_ihts = StateConstants.NUM_RANDOM_POINTS * StateConstants.CHANNELS
        img_iht_size = StateConstants.IMAGE_IHT_SIZE
        self.img_ihts = [tiles.CollisionTable(img_iht_size, "safe") for _ in xrange(num_img_ihts)]

        self.pp_ihts = [tiles.CollisionTable(StateConstants.PP_IHT_SIZE, "safe") for _ in xrange(StateConstants.NUM_PP)]

        self.imu_iht = tiles.CollisionTable(StateConstants.IMU_IHT_SIZE, "safe")
        self.odom_iht = tiles.CollisionTable(StateConstants.ODOM_IHT_SIZE, "safe")

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
        self.last_ir_raw = None

        self.features_to_use = features_to_use

    @timing
    def get_phi(self, image, bump, ir, imu, odom, bias, weights = None,*args, **kwargs):

        phi = np.zeros(StateConstants.TOTAL_FEATURE_LENGTH)

        valid_image = lambda image: image is not None and len(image) > 0 and len(image[0]) > 0

        if 'image' in self.features_to_use:
            # adding image data to state
            if not valid_image(image):
                rospy.logwarn("Image is empty.")
                if self.last_image_raw is not None:
                    image = self.last_image_raw

            if valid_image(image):
                self.last_image_raw = image 
                rgb_points = image[self.pixel_mask].flatten().astype(float)
                rgb_points *= StateConstants.SCALE_RGB
                rgb_inds = np.arange(StateConstants.NUM_RANDOM_POINTS * 3)

                tile_inds = [tiles.tiles(StateConstants.NUM_IMAGE_TILINGS,
                                         self.img_ihts[i],
                                         [rgb_points[i]]) for i in rgb_inds]

                # tile_inds = np.ones((900,4), dtype=int)

                rgb_inds *= StateConstants.IMAGE_IHT_SIZE

                indices = (tile_inds + rgb_inds[:, np.newaxis]).flatten()
                phi[indices] = True

        if 'image_pairs' in self.features_to_use:
            # check if there is an image
            if valid_image(image):

                # save the valid image
                self.last_image_raw = image 

                # get vector of pixels with each pixel=(Channel1,Channel2,...)
                num_channels = StateConstants.CHANNELS
                pixels = image[self.pixel_mask].reshape(-1, num_channels)

                # get vector of L2 norms of above pixels
                norms = np.linalg.norm(pixels, axis=1)

                # find indices to multiply to get upper triangle
                # of the outer product of the arrays
                row, col = np.triu_indices(num_channels, 1)

                # calculate upper triangle of outer product: pixels, pixels
                dots = np.einsum('ij,ij->i', pixels[row], pixels[col])

                # calculate upper triangle of outer product: norms, norms
                norm_product = np.einsum('i,i->i', norms[row], norms[col])

                # find cosine similarity
                cos_sim = dots/norm * StateConstants.SCALE_PP
                assert cos_sim.size == StateConstants.NUM_PP

                # get indices form tile coding
                pp_inds = np.arange(StateConstants.NUM_PP)
                tile_inds = [tiles.tiles(StateConstants.NUM_PP_TILINGS,
                                         self.pp_ihts[i],
                                         [cos_sim[i]]) for i in pp_inds]

                # offset tilecoding for each pixel by the IHT size to map
                # each pixel to a different set of PP_IHT_SIZE indices
                pp_inds *= StateConstants.PP_IHT_SIZE
                indices = (tile_inds + pp_inds[:, np.newaxis]).flatten()

                # offset all indices to correspond to the Pixel pairs
                # section of the feature array
                indices += StateConstants.PP_START_INDEX
                phi[indices] = True

        if 'imu' in self.features_to_use:
            if imu is None:
                rospy.logwarn("No imu value.")
                if self.last_imu_raw is not None:
                    imu = self.last_imu_raw

            if imu is not None:
                self.last_imu_raw = imu

                indices = np.array(tiles.tiles(StateConstants.NUM_IMU_TILINGS,
                                                self.imu_iht, 
                                                [imu*StateConstants.SCALE_IMU]))

                phi[indices + StateConstants.IMU_START_INDEX] = True

        if 'odom' in self.features_to_use:
            if odom is None:
                rospy.logwarn("No odom value.")
                if self.last_odom_raw is not None:
                    odom = self.last_odom_raw

            if odom is not None:
                self.last_odom_raw = odom

                indices = np.array(tiles.tiles(StateConstants.NUM_ODOM_TILINGS,
                                                self.odom_iht,
                                                (odom * StateConstants.SCALE_ODOM).tolist(),
                                                []))

                phi[indices + StateConstants.ODOM_START_INDEX] = True

        if 'ir' in self.features_to_use:
            if ir is None:
                rospy.logwarn("No ir value.")
                if self.last_ir_raw is not None:
                    ir = self.last_ir_raw

            if ir is not None:
                self.last_ir_raw = ir
                # indices = np.asarray(ir)
                # indices += np.array([0,64,128])
                ir_1 = [int(x) for x in format(ir[0], '#08b')[2:]]
                ir_2 = [int(x) for x in format(ir[1], '#08b')[2:]]
                ir_3 = [int(x) for x in format(ir[2], '#08b')[2:]]
                value = ir_1 + ir_2 + ir_3
                indices = np.nonzero(value)[0]

                phi[np.asarray(indices) + StateConstants.IR_START_INDEX] = True

        # bump
        if 'bump' in self.features_to_use and bump is not None:
            phi[StateConstants.indices_in_phi['bump']] = bump

        # bias unit
        if 'bias' in self.features_to_use:
            phi[StateConstants.indices_in_phi['bias']] = True

        return phi

    def get_observations(self, bump, ir, charging, **kwargs):
        observations = {'bump': bump if bump else (0,0,0),
                          'ir': ir if ir else (0,0,0),
                          'charging': charging if charging else False,
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
