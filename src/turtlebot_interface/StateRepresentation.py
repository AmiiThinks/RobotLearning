import random
import tiles3

"""
Picks 300 random rgb values from an image and tiles those 
values to obtain the state representation
"""

DIFF_BW_R = 20
DIFF_BW_G = 20
DIFF_BW_B = 20
DIFF_BW_BUMP = 1

class StateRepresentation:
    def __init__(self):
        self.iht = IHT(1000000)

    # image: a 2D array with 
    def Random300Points(image):
        if image == None or len(image) == 0 or len(image[0] == 0):
            return []

        random_points = []
        
        for p in range(300):
            p1 = random.randint(0, len(image))
            p2 = random.randint(0, len(image[0]))

            random_points.append(image[p1][p2])

        return random_points

    def GetStateRepresentation(points, action):
        state_representation_raw = []

        rgbpoints_raw = [sp for p in points for sp in p]
        rgb_mod = 0
        for p in range(len(rgbpoints_raw)):
            if (rgb_mod == 0):
                # NOTE: THIS '/' IS ASSUMING PYTHON3 FUNCTIONALITY 
                state_representation_raw.append(p / DIFF_BW_R)
            elif (rgb_mod == 1):
                state_representation_raw.append(p / DIFF_BW_G)
            elif (rgb_mod == 2):
                state_representation_raw.append(p / DIFF_BW_B)

            rgb_mod = (rgb_mod + 1) % 3

        # TODO: ADD BUMP SENSOR DATA TO STATE REPRESENTATION
        # TODO: ADD ACTION TO THE STATE REPRESENTATION
        # TODO: PROCESS ACTION APPROPRIATELY
        
        return tiles(iht, state_representation_raw, [action] ) 

    
