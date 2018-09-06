from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	#vs = cv2.VideoCapture(0)
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
 
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
 
# initialize the FPS throughput estimator
fps = FPS().start()

#img1 = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('obj.png', cv2.IMREAD_GRAYSCALE)

brsk = cv2.BRISK_create()
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# find the keypoints and descriptors with ORB
#kp1, des1 = brsk.detectAndCompute(img1,None)
kp2, des2 = brsk.detectAndCompute(img2,None)

while True :
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
            break

    (H, W) = frame.shape[:2]
    
    kp1, des1 = brsk.detectAndCompute(frame,None)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    #print(len(matches))
    if matches[0].distance < 80 and len(matches)>200 :
        x_coords = [int(kp1[matches[i].queryIdx].pt[0]) for i in range(0,int(len(matches)/10))]
        y_coords = [int(kp1[matches[i].queryIdx].pt[1]) for i in range(0,int(len(matches)/10))]
    
        cv2.rectangle(frame, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), (0, 255, 0), 2)

    # update the FPS counter
    fps.update()
    fps.stop()
    
    # initialize the set of information we'll be displaying on
    # the frame
    info = ['r[0]', "{:.2f}".format(fps.fps())]

    # loop over the info tuples and draw them on our frame
    cv2.putText(frame, info[1], (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
     
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
 
# otherwise, release the file pointer
else:
	vs.release()
 
# close all windows
cv2.destroyAllWindows()     






