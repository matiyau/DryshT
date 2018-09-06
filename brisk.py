import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('obj.png', cv2.IMREAD_GRAYSCALE)

brsk = cv2.BRISK_create()

# find the keypoints and descriptors with ORB
kp1, des1 = brsk.detectAndCompute(img1,None)
kp2, des2 = brsk.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

x_coords = [int(kp1[matches[i].queryIdx].pt[0]) for i in range(0,int(len(matches)/2))]
y_coords = [int(kp1[matches[i].queryIdx].pt[1]) for i in range(0,int(len(matches)/2))]

cv2.rectangle(img1, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), (0, 255, 0), 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=2)
plt.imshow(img3),plt.show()







