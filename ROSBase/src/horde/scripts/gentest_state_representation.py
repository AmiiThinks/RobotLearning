import numpy as np
import rospy
import time

from state_representation import StateConstants, StateManager
from tools import overrides, timing

class GenTestStateConstants:
	MIN_ALIVE_TIME = 150
	NUM_POINTS_REPLACE = 60

class GenTestStateManager(StateManager):
	def __init__(self, features_to_use):
		super(GenTestStateManager, self).__init__(features_to_use)
		self.pixel_time_alive = np.zeros(StateConstants.NUM_RANDOM_POINTS)

	@overrides(StateManager)
	@timing
	def get_phi(self, image, bump, ir, imu, odom, bias, weights = None):
		# update weights to reflect removal of features
		if (weights is not None):
			pixel_weights = np.zeros(StateConstants.NUM_RANDOM_POINTS)

			for pixel in range(StateConstants.NUM_RANDOM_POINTS):
				pixel_weights[pixel] = weights[StateConstants.PIXEL_FEATURE_LENGTH * pixel + StateConstants.IMAGE_START_INDEX: 
				                               StateConstants.PIXEL_FEATURE_LENGTH * (pixel + 1) + StateConstants.IMAGE_START_INDEX].sum()

			sorted_pixels = np.absolute(pixel_weights).argsort()[:GenTestStateConstants.NUM_POINTS_REPLACE]
			# retrieve the indices of the lowest pixels
			lowest_pixels = [pixel for pixel in sorted_pixels if self.pixel_time_alive[pixel] >= GenTestStateConstants.MIN_ALIVE_TIME]

			random_points = np.random.randint(low=0, 
                                              high=StateConstants.IMAGE_LI*StateConstants.IMAGE_CO, 
                                              size=len(lowest_pixels))

			rand_index = 0
			for pixel in lowest_pixels:
				new_random_point = random_points[rand_index]
				rand_index += 1

				# ensure uniqueness of new random points (shouldn't happen often)
				while new_random_point in self.chosen_indices:
					new_random_point = np.random.randint(low=0,
								                         high=StateConstants.IMAGE_LI*StateConstants.IMAGE_CO,
								                         size=1)

				self.chosen_indices[pixel] = new_random_point
				self.pixel_time_alive[pixel] = 0

				# zero out the appropriate weights
				weights[StateConstants.PIXEL_FEATURE_LENGTH * pixel + StateConstants.IMAGE_START_INDEX: 
				        StateConstants.PIXEL_FEATURE_LENGTH * (pixel + 1) + StateConstants.IMAGE_START_INDEX] = 0   

		
        # get the new state
		feature_state = super(GenTestStateManager, self).get_phi(image, bump, ir, imu, odom, bias, None)

		self.pixel_time_alive += 1 # increment time alive	

		return feature_state
