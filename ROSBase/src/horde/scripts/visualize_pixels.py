#!/usr/bin/env python

from PIL import Image
import csv
from tools import timing

class Visualize():
    def __init__(self,points,imsizex,imsizey):

        self.voronoi={(p[0],p[1]):[] for p in points}
        for x in range(imsizex):
            for y in range(imsizey):
                mini=imsizex*imsizex+imsizey*imsizey
                for p in points:
                    dist = (p[0]-x)*(p[0]-x)+(p[1]-y)*(p[1]-y)
                    if dist < mini:
                        mini=dist
                        fp=p
                self.voronoi[(fp[0],fp[1])].append([x,y])

    @timing
    def update_colours(self,image):
    
        # Create blank image
        im=Image.new("RGB",(len(image),len(image[0])))
        pix = im.load()
        
        # Open and read CSV file
        # f=open("rgb.csv","rb")
        # colours=csv.reader(f)
        
        # Save read values to a matrix
        # colours2=[[0 for x in range(3)] for y in range(3)]
        # i=0
        # for colour in colours:
        #     colours2[i][0]=colour[0]
        #     colours2[i][1]=colour[1]
        #     colours2[i][2]=colour[2]
        #    i=i+1
        
        # Assign the values to the image
        # This part can be changed to rearrange what the image looks like.
        
        # set each pixel to be the colour of the closest point (voronoi)

        for r in self.voronoi:
            for p in self.voronoi[r]:
                pix[p[0],p[1]]=(int(image[r[0]][r[1]][0]),int(image[r[0]][r[1]][1]),int(image[r[0]][r[1]][2]))


        # f.close()

        # Save the image
        im.save("visualization.png", "PNG")
