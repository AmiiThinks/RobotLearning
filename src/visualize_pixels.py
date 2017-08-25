"""
Michele Albach, Niko Yasui
June 8th 2017

Visualizer:

initializes by creating a Voronoi diagram to match the image size
and set of points

then update_colours can be called with image data to properly 
colorize the Voronoi diagram

the Voronoi diagram is plotted using matplotlib
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rospy

import tools
from tools import timing

from state_representation import StateConstants

# turn off toolbar
matplotlib.rcParams['toolbar'] = 'None'


class Visualize:
    def __init__(self, mask, imsizex, imsizey, dpi=100):

        # convert mask to x,y points
        points = map(list, zip(*reversed(map(list, np.where(mask)))))

        # make subdivision 
        rect = (0, 0, imsizex, imsizey)
        self.subdiv = cv2.Subdiv2D(rect)
        self.subdiv.insert(points)

        # initialize figure
        self.fig = plt.figure("Image Stream",
                              figsize=(imsizex / dpi, imsizey / dpi),
                              dpi=dpi)

        # initialize ax
        self.ax = self.fig.add_axes([0, 0, 1, 1], frame_on=False)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        # initialize image
        self.im = self.ax.imshow(np.zeros((imsizey, imsizex)),
                                 interpolation='none',
                                 animated=True)

        self.image = None
        # start figure
        self.fig.show()
        self.fig.canvas.draw()

    @timing
    def update_colours(self):
        if self.image is None:
            return None

        (facets, centers) = self.subdiv.getVoronoiFacetList([])
        img = np.zeros(self.image.shape, dtype=self.image.dtype)

        # color image
        for i in range(len(facets)):
            ifacet = np.array([f for f in facets[i]], np.int)
            color = self.image[int(centers[i][1])][int(centers[i][0])].tolist()
            cv2.fillConvexPoly(img, ifacet, color)

        # update image data
        self.im.set_data(np.flipud(np.fliplr(img)))

        # draw image
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def update_image(self, image_data):
        self.image = np.fromstring(image_data.data,
                          np.uint8).reshape(480, 640, 3) 

if __name__ == "__main__":

    rospy.init_node('visualizer', anonymous=True)
    
    # setup visualization
    num_pixels = StateConstants.IMAGE_LI * StateConstants.IMAGE_CO
    num_chosen = StateConstants.NUM_RANDOM_POINTS
    chosen_indices = np.random.choice(a=num_pixels,
                                           size=num_chosen,
                                           replace=False)

    pixel_mask = np.zeros(num_pixels, dtype=np.bool)
    pixel_mask[chosen_indices] = True
    pixel_mask = pixel_mask.reshape(StateConstants.IMAGE_LI,
                                    StateConstants.IMAGE_CO)
    rospy.loginfo("Creating visualization.")
    visualization = Visualize(pixel_mask,
                              imsizex=640,
                              imsizey=480)
    rospy.loginfo("Done creatiing visualization.")
    
    # setup image subscriber
    rospy.Subscriber("/camera/rgb/image_rect_color",
                     tools.topic_format["/camera/rgb/image_rect_color"],
                     visualization.update_image)    

    r = rospy.Rate(int(1.0 / 0.1))

    while not rospy.is_shutdown():
        visualization.update_colours()    
        r.sleep()


# if __name__ == "__main__":

#     rospy.init_node('visualizer', anonymous=True)
    
#     # set up dictionary to receive image
#     recent = {'image': Queue(1)}

#     # setup sensor parsers
#     rospy.Subscriber('image',
#                      tools.topic_format['image'],
#                      recent['image'].put)    

#     r = rospy.Rate(int(1.0 / 0.1))
    


#     num_pixels = StateConstants.IMAGE_LI * StateConstants.IMAGE_CO
#     num_chosen = StateConstants.NUM_RANDOM_POINTS
#     chosen_indices = np.random.choice(a=num_pixels,
#                                            size=num_chosen,
#                                            replace=False)

#     pixel_mask = np.zeros(num_pixels, dtype=np.bool)
#     pixel_mask[chosen_indices] = True
#     pixel_mask = pixel_mask.reshape(StateConstants.IMAGE_LI,
#                                     StateConstants.IMAGE_CO)
#     rospy.loginfo("Creating visualization.")
#     visualization = Visualize(pixel_mask,
#                               imsizex=640,
#                               imsizey=480)
#     rospy.loginfo("Done creatiing visualization.")
    
#     while not rospy.is_shutdown():
#         visualization.update_colours(recent['image'].get_nowait())    
#         r.sleep()
