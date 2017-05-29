from PIL import Image
import csv

def updateColours():
    # Open and read CSV file
    f=open("rgb.csv","rb")
    colours=csv.reader(f)

    # Save read values to a matrix
    colours2=[[0 for x in range(3)] for y in range(3)]
    i=0
    for colour in colours:
        colours2[i][0]=colour[0]
        colours2[i][1]=colour[1]
        colours2[i][2]=colour[2]
        i=i+1

    # Assign the values to the image
    # This part can be changed to rearrange what the image looks like. Change
    #    the values in the range() functions. It must be known how many lines
    #    were read from the csv file and what order they go in.
    for x in range(480): 
        for y in range(160):
            pix[x,y] = (int(colours2[0][0]),int(colours2[0][1]),int(colours2[0][2]))
        for y in range(161, 320):
            pix[x,y] = (int(colours2[1][0]),int(colours2[1][1]),int(colours2[1][2]))
        for y in range(321, 480):
            pix[x,y] = (int(colours2[2][0]),int(colours2[2][1]),int(colours2[2][2]))

    f.close()

    # Save the image
    im.save("visualization.png", "PNG")


