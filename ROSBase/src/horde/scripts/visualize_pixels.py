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

from tools import timing

# turn off toolbar
matplotlib.rcParams['toolbar'] = 'None'

class Visualize():
    def __init__(self, mask, imsizex, imsizey, dpi=100):

        # convert mask to x,y points
        points = map(list, zip(*reversed(map(list, np.where(mask)))))

        # make subdivision
        rect = (0, 0, imsizex, imsizey)
        self.subdiv = cv2.Subdiv2D(rect)
        self.subdiv.insert(points)

        # initialize figure
        self.fig = plt.figure("Image Stream",
                              figsize=(imsizex/dpi, imsizey/dpi),
                              dpi=dpi)

        # initialize ax
        self.ax = self.fig.add_axes([0, 0, 1, 1], frame_on=False)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

        # initialize image
        self.im = self.ax.imshow(np.zeros((imsizey,imsizex)),
                                 interpolation='none',
                                 animated=True)

        # start figure
        self.fig.show()
        self.fig.canvas.draw()


    @timing
    def update_colours(self, image):
        if image is None: return None

        (facets, centers) = self.subdiv.getVoronoiFacetList([])
        img = np.zeros(image.shape, dtype=image.dtype)

        # color image
        for i in range(len(facets)):
            ifacet_arr = []
            for f in facets[i]:
                ifacet_arr.append(f)
             
            ifacet = np.array(ifacet_arr, np.int)
            color = image[int(centers[i][0])][int(centers[i][1])].tolist()
            cv2.fillConvexPoly(img, ifacet, color)

        # update image data
        self.im.set_data(img)

        # draw image
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()