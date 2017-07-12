import numpy as np
import rospy

from state_representation import StateConstants, StateManager
from tools import overrides

class GenTestStateConstants:
	MIN_ALIVE_TIME = 100
	NUM_POINTS_REPLACE = 30

class GenTestStateManager(StateManager):
	def __init__(self):
		super(GenTestStateManager, self).__init__()
		self.reset_mask = np.ones(StateConstants.TOTAL_FEATURE_LENGTH)
		self.pixel_time_alive = np.zeros(StateConstants.NUM_RANDOM_POINTS)

	@overrides(StateManager)
	def get_state_representation(self, image, bumper_information, action, weights = None):
		reset_mask = np.ones(StateConstants.TOTAL_FEATURE_LENGTH)

		rospy.loginfo("WEIGHTS")
		rospy.loginfo(weights)
		# update state appropriately here then call the original
		if (weights is not None):
			pixel_weights = np.zeros(StateConstants.NUM_RANDOM_POINTS)

			for w in range(StateConstants.NUM_RANDOM_POINTS):
				for rgb in range(3):
					pixel_weights[w] += weights[3 * w + rgb + StateConstants.NUM_BUMPERS]

			lowest_pixels = pixel_weights.argsort()[:GenTestStateConstants.NUM_POINTS_REPLACE]

			for pixel in lowest_pixels:
				if (self.pixel_time_alive[pixel] >= GenTestStateConstants.MIN_ALIVE_TIME):
					new_random_point = self.random_points(1)[0]
					while new_random_point not in self.chosen_points:
						new_random_point = self.random_points(1)[0]
					self.chosen_points[pixel] = new_random_point

					for rgb in range(3):
						reset_mask[pixel * 3 + i + StateConstants.NUM_BUMPERS] = 0

		feature_state = super(GenTestStateManager, self).get_state_representation(
			image, bumper_information, action)
		
		self.pixel_time_alive += np.ones(StateConstants.NUM_RANDOM_POINTS) # increment time alive
		
		return {"reset_mask": reset_mask, "feature_state": feature_state}