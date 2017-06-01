import visualize_pixels

# Continuously calls the function from visualizePixels.py
# To use, first open the image (if it exists yet) using 'eog visualization.png',
#    then run this script. If the file 'rgb.csv' changes while this script is
#    running, the image will be updated.

while(1):
    visualizePixels.updateColours()
