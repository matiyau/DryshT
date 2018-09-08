from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import darknet as dn

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

dn_dir = "/home/n7/darknet/"
net = dn.load_net(dn_dir + "cfg/yolov3.cfg", dn_dir + "yolov3.weights", 0)
meta = dn.load_meta(dn_dir + "cfg/coco.data")

#img1 = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('obj.png', cv2.IMREAD_GRAYSCALE)

brsk = cv2.xfeatures2d.SIFT_create()
# create BFMatcher object
bf = cv2.BFMatcher_create()

# find the keypoints and descriptors with ORB
#kp1, des1 = brsk.detectAndCompute(img1,None)
#kp2, des2 = brsk.detectAndCompute(img2,None)

init = False
kp2=0
des2=0

while True :
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
            break

    (H, W) = frame.shape[:2]

    cv2.imwrite("f.png",frame)
        
    r = dn.detect(net, meta, "f.png")

    for i in r :
        #print(i[0])
        if i[0] == "bottle" :
            [xa, ya, xb, yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
            [xa, ya, xb, yb] = [max(0, xa), max(0,ya), min(xb, W), min(yb, H)]
            focus = frame[ya:yb, xa:xb]
            #print(xa,ya,xb,yb)
            if len(focus) == 0 :
                continue
            kp1, des1 = brsk.detectAndCompute(focus,None)

            if not init :
                kp12, des2 = kp1,des1
                init = True
                continue
                
            # Match descriptors.
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            #print(len(matches),len(good))
            matches = good
            #print(good)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            print(len(matches))
            #print(float(len(matches))/min(len(des1),len(des2)))
            #and matches[0].distance < 100 and matches[len(matches)/3].distance < 200
            if len(matches)>0 and matches[-1].distance<350 :
                print(matches[0].distance, matches[-1].distance)
                #x_coords = [int(kp1[matches[i].queryIdx].pt[0]) for i in range(0,int(len(matches)/10))]
                #y_coords = [int(kp1[matches[i].queryIdx].pt[1]) for i in range(0,int(len(matches)/10))]
                kp2, des2 = kp1, des1
        
                #cv2.rectangle(frame, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), (0, 255, 0), 2)
                cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)

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






