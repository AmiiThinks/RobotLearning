"""
Michele Albach
June 8th 2017

Visualizer:

initializes by creating a voronoi diagram to match the image size
and set of points

then update_colours can be called with image data to properly 
colorize the voronoi diagram

updates an image called visualization.png which will update 
automatically with each change (even when open)
"""

from PIL import Image
import csv
from tools import timing

class Visualize():
    # Note: creating the diagram takes some time, but should only
    #       need to be done once when the random points are chosen
    def __init__(self,points,imsizex,imsizey):

        # the diagram is saved as a dictionary
        self.voronoi={(p[0],p[1]):[] for p in points}

        # for every pixel in the image:
        for x in range(imsizex):
            for y in range(imsizey):
                # check every point to see if it's closer than the
                #      closest so far
                mini=imsizex*imsizex+imsizey*imsizey
                for p in points:
                    dist = (p[0]-x)*(p[0]-x)+(p[1]-y)*(p[1]-y)
                    if dist < mini:
                        mini=dist
                        fp=p
                # save the pixel into the value of the closest point
                self.voronoi[(fp[0],fp[1])].append([x,y])

    # update_colours: goes through each region in the voronoi diagram and
    #      colours all pixels in that region the colour of the point
    @timing
    def update_colours(self,image):
    
        # Create blank image
        im=Image.new("RGB",(len(image),len(image[0])))
        pix = im.load()
        
        # for each pixel in each region, colour it to match the region's point
        for r in self.voronoi:
            for p in self.voronoi[r]:
                pix[p[0],p[1]]=(int(image[r[0]][r[1]][0]),int(image[r[0]][r[1]][1]),int(image[r[0]][r[1]][2]))


        # Save the image
        im.save("visualization.png", "PNG")
