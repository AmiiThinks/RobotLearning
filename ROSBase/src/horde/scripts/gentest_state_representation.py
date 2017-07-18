import numpy as np
import rospy
import timeit

from state_representation import StateConstants, StateManager
from tools import overrides, timing

class GenTestStateConstants:
	MIN_ALIVE_TIME = 100
	NUM_POINTS_REPLACE = 60

class GenTestStateManager(StateManager):
	def __init__(self):
		super(GenTestStateManager, self).__init__()
		self.pixel_time_alive = np.zeros(StateConstants.NUM_RANDOM_POINTS)

	@overrides(StateManager)
	@timing
	def get_phi(self, image, weights = None):
		# update weights to reflect removal of features

		if (weights is not None):
			pixel_weights = np.zeros(StateConstants.NUM_RANDOM_POINTS)

			for w in range(StateConstants.NUM_RANDOM_POINTS):
				for rgb in range(3 * StateConstants.NUM_FEATURES_PER_COL_VAL):
					pixel_weights[w] += weights[3 * StateConstants.NUM_FEATURES_PER_COL_VAL * w + rgb]

			# retrieve the indices of the lowest pixels
			lowest_pixels = pixel_weights.argsort()[:GenTestStateConstants.NUM_POINTS_REPLACE]

			for pixel in lowest_pixels:
				if (self.pixel_time_alive[pixel] >= GenTestStateConstants.MIN_ALIVE_TIME):
					new_random_point = np.random.choice(a=StateConstants.IMAGE_LI*StateConstants.IMAGE_CO,
                                          				size=1, 
                                          				replace=False)[0]
					while new_random_point in self.chosen_indices:
						new_random_point = np.random.choice(a=StateConstants.IMAGE_LI*StateConstants.IMAGE_CO,
                                          					size=1, 
                                          					replace=False)[0]
					self.chosen_indices[pixel] = new_random_point

					for index in range(3 * StateConstants.NUM_FEATURES_PER_COL_VAL):
						weights[3 * pixel * StateConstants.NUM_FEATURES_PER_COL_VAL + index] = 0
			
		# get the new state
		feature_state = super(GenTestStateManager, self).get_phi(image)
		
		self.pixel_time_alive += np.ones(StateConstants.NUM_RANDOM_POINTS) # increment time alive
		
		return feature_state