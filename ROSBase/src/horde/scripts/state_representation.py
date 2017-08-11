#!/usr/bin/env python

"""
Author: Niko Yasui, Shibhansh Dohare, Parash Rahman, Michele Albach,
David Quail June 1, 2017.

"""
import random

import numpy as np
import rospy
from scipy.misc import comb

from CTiles import tiles
from tools import get_next_pow2, timing

# np.set_printoptions(threshold=np.nan)

"""
# StateManager:

Picks NUM_RANDOM_POINTS random rgb values from an image and tiles those
values to obtain the state representation
"""


class StateConstants:
    # image tiles
    NUM_RANDOM_POINTS = 50
    CHANNELS = 3
    NUM_IMAGE_TILINGS = 4
    NUM_IMAGE_INTERVALS = 4
    SCALE_RGB = NUM_IMAGE_INTERVALS / 256.0
    IMAGE_IHT_SIZE = get_next_pow2(
            (NUM_IMAGE_INTERVALS + 1) * NUM_IMAGE_TILINGS)
    PIXEL_FEATURE_LENGTH = CHANNELS * IMAGE_IHT_SIZE
    TOTAL_IMAGE_FEATURE_LENGTH = NUM_RANDOM_POINTS * PIXEL_FEATURE_LENGTH
    IMAGE_START_INDEX = 0

    # constants relating to image size recieved
    IMAGE_LI = 480  # rows
    IMAGE_CO = 640  # columns

    # IMU tiles
    NUM_IMU_TILINGS = 4
    NUM_IMU_TILES = 4
    SCALE_IMU = NUM_IMU_TILES / 2.0  # range is [-1, 1]
    IMU_IHT_SIZE = get_next_pow2((NUM_IMU_TILES + 1) * NUM_IMU_TILINGS)
    IMU_START_INDEX = IMAGE_START_INDEX + TOTAL_IMAGE_FEATURE_LENGTH

    # Odom tiles
    NUM_ODOM_TILINGS = 1
    NUM_ODOM_TILES = 5
    SCALE_ODOM = NUM_ODOM_TILES / 2.0  # range is [0, 2]
    ODOM_IHT_SIZE = get_next_pow2(
            (NUM_ODOM_TILES + 1) * (NUM_ODOM_TILES + 1) * NUM_ODOM_TILINGS)
    ODOM_START_INDEX = IMU_START_INDEX + IMU_IHT_SIZE

    # IR tiles
    IR_START_INDEX = ODOM_START_INDEX + ODOM_IHT_SIZE
    # IR_ITH_SIZE = 64*3
    # IR_ITH_SIZE = 6*3
    IR_ITH_SIZE = 6
    # IR_ITH_SIZE = 3

    # pixel pairs
    NUM_PP = comb(NUM_RANDOM_POINTS, 2, exact=True)
    NUM_PP_TILINGS = 4
    NUM_PP_TILES = 4
    SCALE_PP = NUM_PP_TILES / 2.  # [-1, 1]
    PP_IHT_SIZE = get_next_pow2((NUM_PP_TILES + 1) * NUM_PP_TILINGS)
    PP_FEATURE_LENGTH = NUM_PP * PP_IHT_SIZE
    PP_START_INDEX = IR_START_INDEX + IR_ITH_SIZE

    # the 1 represents the bias unit, 3 for bump
    TOTAL_FEATURE_LENGTH = (TOTAL_IMAGE_FEATURE_LENGTH + IMU_IHT_SIZE +
                            ODOM_IHT_SIZE + IR_ITH_SIZE + 3 + 1 +
                            PP_FEATURE_LENGTH)

    indices_in_phi = {
        'image': np.arange(IMAGE_START_INDEX,
                           IMAGE_START_INDEX + TOTAL_IMAGE_FEATURE_LENGTH),
        'cimage': np.arange(IMAGE_START_INDEX,
                            IMAGE_START_INDEX + TOTAL_IMAGE_FEATURE_LENGTH),
        'imu': np.arange(IMU_START_INDEX, IMU_START_INDEX + IMU_IHT_SIZE),
        'odom': np.arange(ODOM_START_INDEX, ODOM_START_INDEX + ODOM_IHT_SIZE),
        'ir': np.arange(IR_START_INDEX, IR_START_INDEX + IR_ITH_SIZE),
        'pixel_pairs': np.arange(PP_START_INDEX,
                                 PP_START_INDEX + PP_FEATURE_LENGTH),
        'bump': np.arange(PP_START_INDEX + PP_FEATURE_LENGTH,
                          PP_START_INDEX + PP_FEATURE_LENGTH + 3),
        'bias': np.array([TOTAL_FEATURE_LENGTH - 1]),
        'last_action': np.array([], dtype=int),
    }

    num_active_features = {
        "image": NUM_RANDOM_POINTS * CHANNELS * NUM_IMAGE_TILINGS,
        'cimage': NUM_RANDOM_POINTS * CHANNELS * NUM_IMAGE_TILINGS,
        'imu': NUM_IMU_TILINGS,
        'odom': NUM_ODOM_TILINGS,
        'ir': 3,
        'bump': 3,
        'bias': 1,
        'last_action': 1,
        'pixel_pairs': NUM_PP * NUM_PP_TILINGS
    }


class StateManager(object):
    def __init__(self, features_to_use):

        num_img_ihts = StateConstants.NUM_RANDOM_POINTS * \
                       StateConstants.CHANNELS
        img_iht_size = StateConstants.IMAGE_IHT_SIZE
        self.img_ihts = [tiles.CollisionTable(img_iht_size, "safe") for _ in
                         range(num_img_ihts)]

        self.pp_ihts = [
            tiles.CollisionTable(StateConstants.PP_IHT_SIZE, "safe") for _ in
            range(StateConstants.NUM_PP)
        ]

        self.imu_iht = tiles.CollisionTable(StateConstants.IMU_IHT_SIZE,
                                            "safe")
        self.odom_iht = tiles.CollisionTable(StateConstants.ODOM_IHT_SIZE,
                                             "safe")

        # set up mask to chose pixels
        num_pixels = StateConstants.IMAGE_LI * StateConstants.IMAGE_CO
        num_chosen = StateConstants.NUM_RANDOM_POINTS
        self.chosen_indices = np.random.choice(a=num_pixels,
                                               size=num_chosen,
                                               replace=False)
        self.pixel_mask = np.zeros(num_pixels, dtype=np.bool)
        self.pixel_mask[self.chosen_indices] = True
        self.pixel_mask = self.pixel_mask.reshape(StateConstants.IMAGE_LI,
                                                  StateConstants.IMAGE_CO)

        self.last_image_raw = np.zeros((StateConstants.IMAGE_LI,
                                        StateConstants.IMAGE_CO,
                                        3))
        self.last_imu_raw = float()
        self.last_odom_raw = np.zeros(4)
        self.last_ir_raw = (0, 0, 0)
        self.last_charging_raw = False
        self.last_bump_raw = False

        self.features_to_use = features_to_use

    @timing
    def get_phi(self, image, bump, ir, imu, odom, bias, weights=None, *args,
                **kwargs):

        phi = np.zeros(StateConstants.TOTAL_FEATURE_LENGTH, dtype=bool)

        # phi = np.zeros(2)
        # self.features_to_use = set()
        # phi[0] = imu if imu is not None else 0
        # phi[1] = 1

        def valid_image(img):
            return img is not None and len(img) > 0 and len(img[0]) > 0

        if not valid_image(image):
            image = self.last_image_raw
            if 'image' in self.features_to_use:
                rospy.logwarn("Image is empty.")

        self.last_image_raw = image

        if 'image' in self.features_to_use:
            rgb_points = image[self.pixel_mask].flatten().astype(float)
            rgb_points *= StateConstants.SCALE_RGB
            rgb_inds = np.arange(StateConstants.NUM_RANDOM_POINTS * 3)

            tile_inds = [tiles.tiles(StateConstants.NUM_IMAGE_TILINGS,
                                     self.img_ihts[i],
                                     [rgb_points[i]]) for i in rgb_inds]

            # tile_inds = np.ones((900,4), dtype=int)

            rgb_inds *= StateConstants.IMAGE_IHT_SIZE

            indices = (tile_inds + rgb_inds[:, np.newaxis]).ravel()
            assert np.min(indices) >= StateConstants.IMAGE_START_INDEX
            assert np.max(indices) <= (
                                    StateConstants.IMAGE_START_INDEX +
                                    StateConstants.TOTAL_IMAGE_FEATURE_LENGTH
                                    )
            phi[indices] = 1

        if 'pixel_pairs' in self.features_to_use:
            # get vector of pixels with each pixel=(Channel1,Channel2,...)
            num_channels = StateConstants.CHANNELS
            pixels = image[self.pixel_mask].reshape(-1, num_channels)

            # get vector of L2 norms of above pixels
            norms = np.linalg.norm(pixels, axis=1)
            assert norms.size == pixels.shape[0]

            # find indices to multiply to get upper triangle
            # of the outer product of the arrays
            row, col = np.triu_indices(norms.size, 1)

            # calculate upper triangle of outer product: pixels, pixels
            dots = np.einsum('ij,ij->i', pixels[row], pixels[col])

            # calculate upper triangle of outer product: norms, norms
            norm_product = np.einsum('i,i->i', norms[row], norms[col])

            # find cosine similarity
            cos_sim = dots / norm_product
            print(np.min(cos_sim), np.max(cos_sim))
            assert cos_sim.size == StateConstants.NUM_PP
            assert (cos_sim <= 1.0).all() and (cos_sim >= -1.0).all()

            cos_sim *= StateConstants.SCALE_PP

            # get indices form tile coding
            pp_inds = np.arange(StateConstants.NUM_PP)
            tile_inds = [tiles.tiles(StateConstants.NUM_PP_TILINGS,
                                     self.pp_ihts[i],
                                     [cos_sim[i]]) for i in pp_inds]

            # offset tilecoding for each pixel by the IHT size to map
            # each pixel to a different set of PP_IHT_SIZE indices
            pp_inds *= StateConstants.PP_IHT_SIZE
            indices = (tile_inds + pp_inds[:, np.newaxis]).ravel()

            # offset all indices to correspond to the Pixel pairs
            # section of the feature array
            indices += StateConstants.PP_START_INDEX

            assert np.min(indices) >= StateConstants.PP_START_INDEX
            assert np.max(indices) <= (
                                    StateConstants.PP_START_INDEX +
                                    StateConstants.PP_FEATURE_LENGTH
                                    )
            phi[indices] = 1

        if imu is None:
            imu = self.last_imu_raw
            if 'imu' in self.features_to_use:
                rospy.logwarn("No imu value.")

        if 'imu' in self.features_to_use:
            indices = np.array(tiles.tiles(StateConstants.NUM_IMU_TILINGS,
                                           self.imu_iht,
                                           [imu * StateConstants.SCALE_IMU]))

            phi[indices + StateConstants.IMU_START_INDEX] = 1

        if odom is None:
            odom = self.last_odom_raw
            if 'odom' in self.features_to_use:
                rospy.logwarn("No odom value.")

        if 'odom' in self.features_to_use:
            indices = np.array(tiles.tiles(StateConstants.NUM_ODOM_TILINGS,
                                           self.odom_iht,
                                           (
                                               odom *
                                               StateConstants.SCALE_ODOM).tolist(),
                                           []))

            phi[indices + StateConstants.ODOM_START_INDEX] = 1

        if ir is None:
            ir = self.last_ir_raw
            if 'ir' in self.features_to_use:
                rospy.logwarn("No ir value.")

        if 'ir' in self.features_to_use:
            # indices = np.asarray(ir)
            # indices += np.array([0,64,128])

            ir_1 = [int(x) for x in format(ir[0], '#08b')[2:]]
            ir_2 = [int(x) for x in format(ir[1], '#08b')[2:]]
            ir_3 = [int(x) for x in format(ir[2], '#08b')[2:]]
            value = ir_1 + ir_2 + ir_3

            # # if only need the information about the region the robot is (
            # left,center,right)
            # in_right = ir_1[3] | ir_1[0] | ir_2[3] | ir_2[0] | ir_3[3] |
            # ir_3[0]
            # in_left = ir_1[5] | ir_1[1] | ir_2[5] | ir_2[1] | ir_3[5] |
            # ir_3[1]
            # in_center = ir_1[4] | ir_1[2] | ir_2[4] | ir_2[2] | ir_3[4] |
            # ir_3[2]
            # if in_center:
            #     in_left = 0
            #     in_right = 0
            # value = [in_left, in_center, in_right]

            # if only want to use the data from the center IR of the robot
            value = ir_2
            indices = np.nonzero(value)[0]

            phi[np.asarray(indices) + StateConstants.IR_START_INDEX] = 1

        # bump
        if bump is None:
            bump = self.last_bump_raw
            if 'bump' in self.features_to_use:
                rospy.logwarn("No bump value")

        if 'bump' in self.features_to_use:
            phi[StateConstants.indices_in_phi['bump']] = bump

        # bias unit
        if 'bias' in self.features_to_use:
            phi[StateConstants.indices_in_phi['bias']] = 1

        return phi

    def get_observations(self, bump, ir, charging, odom, imu, **kwargs):
        observations = {
    'bump': any(bump) if bump is not None else self.last_bump_raw,
    'ir': ir if ir is not None else self.last_ir_raw,
    'charging': charging if charging is not None else self.last_charging_raw,
    'speed': odom[3] if odom is not None else self.last_odom_raw[3],
    'imu': imu if imu is not None else self.last_imu_raw,
    }

        return observations


# This is a debugging function. It just generates a random image.
@timing
def DEBUG_generate_rand_image():
    dimensions = [1080, 1080]

    return_matrix = [[0 for _ in range(dimensions[1])] for _ in
                     range(dimensions[0])]

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            color = [random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255)]
            return_matrix[i][j] = color

    return return_matrix
