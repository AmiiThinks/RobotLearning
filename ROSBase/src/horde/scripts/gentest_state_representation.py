import numpy as np
import rospy
import time

from state_representation import StateConstants, StateManager
from tools import overrides, timing

class GenTestStateConstants:
	MIN_ALIVE_TIME = 10
	NUM_POINTS_REPLACE = 60

class GenTestStateManager(StateManager):
	def __init__(self):
		super(GenTestStateManager, self).__init__()
		self.pixel_time_alive = np.zeros(StateConstants.NUM_RANDOM_POINTS)

        @timing
	@overrides(StateManager)
	def get_phi(self, image, bumper_status, weights = None):
		# update weights to reflect removal of features

		if (weights is not None):
			pixel_weights = np.zeros(StateConstants.NUM_RANDOM_POINTS)

			for w in range(StateConstants.NUM_RANDOM_POINTS):
				for rgb in range(3 * StateConstants.NUM_FEATURES_PER_COL_VAL):
					pixel_weights[w] += weights[3 * StateConstants.NUM_FEATURES_PER_COL_VAL * w + rgb]

			# retrieve the indices of the lowest pixels
			lowest_pixels = [pixel for pixel in pixel_weights.argsort()[:GenTestStateConstants.NUM_POINTS_REPLACE]
                                         if self.pixel_time_alive[pixel] >= GenTestStateConstants.MIN_ALIVE_TIME]
			
			random_points = np.random.randint(low=0, 
                                                          high=StateConstants.IMAGE_LI*StateConstants.IMAGE_CO, 
                                                          size=len(lowest_pixels))  
			rand_index = 0
			for pixel in lowest_pixels:
				new_random_point = random_points[rand_index]
				rand_index += 1

				while new_random_point in self.chosen_indices:
					new_random_point = np.random.randint(low=0,
								      high=StateConstants.IMAGE_LI*StateConstants.IMAGE_CO,
								      size=1)

				self.chosen_indices[pixel] = new_random_point
				self.pixel_time_alive[pixel] = 0
				for index in range(3 * StateConstants.NUM_FEATURES_PER_COL_VAL):
					weights[3 * pixel * StateConstants.NUM_FEATURES_PER_COL_VAL + index] = 0			
                # get the new state
		feature_state = super(GenTestStateManager, self).get_phi(image, bumper_status)

		self.pixel_time_alive += np.ones(StateConstants.NUM_RANDOM_POINTS) # increment time alive
		
		return feature_state
