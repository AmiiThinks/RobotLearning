import numpy as np
import cv2

img = cv2.imread('visualization.png')
print(img)
print(len(img))
print(len(img[0]))
print(len(img[0][0]))
# 1080 x 1080 x 3
# noice

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img)

cv2.imwrite('sift_keypoints.jpg', img)
