from PIL import Image
import csv
from tools import timing

@timing
def update_colours(points,image):

    # Create blank image
    im=Image.new("RGB",(len(image),len(image[0])))
    pix = im.load()
    print(len(points))
    
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
    for x in range(len(image)):
        for y in range(len(image[0])):
            mini=len(image)*len(image[0])
            for p in range(len(points)):
                dist = (points[p][0]-x)*(points[p][1]-y)
                if dist < mini:
                    mini=dist
                    fp=p
            pix[x,y] = (int(image[points[fp][0]][points[fp][1]][0]),int(image[points[fp][0]][points[fp][1]][1]),int(image[points[fp][0]][points[fp][1]][2]))

    # f.close()

    # Save the image
    im.save("visualization.png", "PNG")


