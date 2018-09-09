from copy import deepcopy as dc
from imutils.video import FPS
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import imutils
import os
import sys
import time

# Set Parameters
matchThresh = 10
mainWindow = "Tracker"

# Select Directories For Darknet Files
scriptDir = sys.path[0]
os.chdir(scriptDir)
sys.path.append(os.path.join(scriptDir, 'dnFiles'))
import darknet as dn
                

# Function To Return The Good Matches
def goodMatches(dsc_src, dsc_dst) :
        #Carry Out FLANN Based Matching
        matches = flann.knnMatch(dsc_src,dsc_dst,k=2)
        
        # Keep Only The Good Matches According To Lowe's Ratio Test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        return good


# Function To Handle Mouse Left Button Release Event
def selectObj(event,x,y,flags,param) :
        if not init :
                if event == cv2.EVENT_LBUTTONUP :
                        global click_x
                        global click_y
                        click_x = x
                        click_y = y
        


# Start The Webcam Stream
print("Starting Webcam...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
 
# Create Objects For Tracker And Its Clone
trkCSRT = cv2.TrackerCSRT_create()
trkCSRT_cp = cv2.TrackerCSRT_create()

# Create Object For SIFT Feature Extractor
sift = cv2.xfeatures2d.SIFT_create()

# Initialize SIFT Parameter Arrays For Various Orientations Of Source Image
kptSIFT_src = [None, None, None, None]
dscSIFT_src = [None, None, None, None]
dims_src = [None,None,None,None]

# Configure The FLANN Based Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Configure YOLO
net = dn.load_net(os.path.join(scriptDir, "dnFiles/yolov3.cfg"), os.path.join(scriptDir, "dnFiles/yolov3.weights"), 0)
meta = dn.load_meta(os.path.join(scriptDir, "dnFiles/coco.data"))
 
# Initialize FPS Estimator
fps = FPS().start()

# Parameter To Store Status Of Tracking
init = False

# Create A Window And Link The Mouse Event To Its Function
cv2.namedWindow('Frame')
cv2.setMouseCallback("Frame", selectObj)

# Position Of Click. Negative Values Indicate No Click
click_x = -1
click_y = -1

while True :
        # Capture Frame From Stream
        frame = vs.read()

        # Check If End Of Stream Has Been Reached
        if frame is None :
                print("Stream Ended")
                break

        # Store Dimensions Of Frame
        H, W = frame.shape[:2]

        # Save The Frame To Memory
        cv2.imwrite("f.png",frame)

        # Run YOLO On The Stored File
        r = dn.detect(net, meta, "f.png")

        # If Tracking Is Not Currently Active
        if not init :
                # Loop Through The Detections Given By YOLO
                for i in r :
                        # Focus Only On The Detections Of Specified Type
                        if i[0] == "cell phone" :
                                # Convert YOLO Coordinates To OpenCV Coordinates
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]

                                # Limit The Coordinates To Image Boundaries
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W-1), min(d_yb, H-1)]

                                # Draw A Rectangle To Show The Detection
                                cv2.rectangle(frame, (d_xa, d_ya), (d_xb, d_yb), (0, 255, 0), 2)

                                # Check If Click Was Inside The Detection Area
                                if click_x > d_xa and click_x < d_xb and click_y > d_ya and click_y < d_yb :
                                        # Set Tracking Status To True
                                        init = True

                                        # Reset Click Position Back To Negative
                                        click_x = -1
                                        click_y = -1

                                        # Initialize CSRT Tracker               
                                        trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))

                                        # Compute SIFT Features For 4 Different Orientations (Separated By 90 Deg.)
                                        roi = cv2.cvtColor(frame[d_ya:d_yb, d_xa:d_xb], cv2.COLOR_BGR2GRAY)
                                        #cv2.imshow("0", roi)
                                        kptSIFT_src[0], dscSIFT_src[0] = sift.detectAndCompute(roi, None)
                                        dims_src[0] = roi.shape[:2]
                                        roi = imutils.rotate_bound(roi, 90)
                                        #cv2.imshow("90", roi)
                                        kptSIFT_src[1], dscSIFT_src[1] = sift.detectAndCompute(roi, None)
                                        dims_src[1] = roi.shape[:2]
                                        roi = imutils.rotate_bound(roi, 90)
                                        #cv2.imshow("180", roi)
                                        kptSIFT_src[2], dscSIFT_src[2] = sift.detectAndCompute(roi, None)
                                        dims_src[2] = roi.shape[:2]
                                        roi = imutils.rotate_bound(roi, 90)
                                        #cv2.imshow("270", roi)
                                        kptSIFT_src[3], dscSIFT_src[3] = sift.detectAndCompute(roi, None)
                                        dims_src[3] = roi.shape[:2]

                                        break

        #If Tracking Is Active               
        else :
                # Clone The Tracker And Use The Clone To Find A Positive
                trkCSRT_cp = trkCSRT
                [resCSRT, roiCSRT] = trkCSRT_cp.update(frame)

                # List To Store Detections Of Particular Class
                dtcts = []

                # Loop Through The Detections Given By YOLO
                for i in r :
                        # Focus Only On The Detections Of Specified Type
                        if i[0] == "cell phone" :
                                dtcts.append(i)
                                
                # If CSRT Tracking Was Successful                
                if resCSRT :
                        for i in dtcts:
                                # Convert YOLO Coordinates To OpenCV Coordinates
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]

                                # Limit The Coordinates To Image Boundaries
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W-1), min(d_yb, H-1)]

                                # Obtain Coordinates Of The Positive Generated By CSRT Tracker
                                [t_xa, t_ya, t_w, t_h] = [int(a) for a in roiCSRT]

                                # Compute Overlap Area Of YOLO Detection With CSRT Positive
                                ovlp_area = (sorted([d_xa, t_xa + t_w, d_xb])[1] - sorted([d_xa,t_xa,d_xb])[1])*(sorted([d_ya,t_ya + t_h, d_yb])[1] - sorted([d_ya,t_ya,d_yb])[1])

                                # Compute YOLO Detection Area
                                dtcn_area = (d_xb - d_xa) * (d_yb - d_ya)

                                # Verify If CSRT Positive Is Actually A Positive
                                if ovlp_area >= 0.5*dtcn_area :
                                        # Draw Rectangle To Show Tracking Of Object
                                        cv2.rectangle(frame, (t_xa, t_ya), (t_xa + t_w, t_ya + t_h), (0, 0, 255), 2)
                                        
                                        # Re-initialize The Tracker On The Current Positive
                                        trkCSRT = cv2.TrackerCSRT_create()
                                        trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))

                                        # Print The Coordinates Of Centre Of Positive
                                        #print("Centre : " + str(t_xa + t_w/2) + ", " + str(t_ya + t_h/20))
                                        break

                                # If No True Positive Is Found, Set resCSRT To False
                                if dtcts.index(i) == len(dtcts)-1 :
                                        resCSRT = False
            

                # If CSRT Tracking Was Unsuccessful Or If A False Positive Was Generated
                if not resCSRT :
                        # If At Least One YOLO Detection Is Found
                        if len(dtcts) :
                                # Compute SIFT Features For Current Frame
                                kptSIFT_dst, dscSIFT_dst = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
                                
                        for i in dtcts :
                                # Convert YOLO Coordinates To OpenCV Coordinates
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]

                                # Limit The Coordinates To Image Boundaries
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W-1), min(d_yb, H-1)]
                                
                                # Generate Matches For Current Frame With Each Of The Initiallly Calculated 4 Orientations
                                good = [None,None,None,None]
                                good[0] = goodMatches(dscSIFT_src[0], dscSIFT_dst)
                                good[1] = goodMatches(dscSIFT_src[1], dscSIFT_dst)
                                good[2] = goodMatches(dscSIFT_src[2], dscSIFT_dst)
                                good[3] = goodMatches(dscSIFT_src[3], dscSIFT_dst)

                                # Fing The Best Matching Set
                                best_index = 0
                                for index in range (1,4) :
                                        if len(good[index]) > len(good[best_index]) :
                                                best_index = index
                                matches_best = good[best_index]

                                # If Best Match Is Better Than The Threshold                                                
                                if len(matches_best) > matchThresh :
                                        # Create NumPy Arrays Of Matching Points In Source And Destination Images
                                        src_pts = np.float32([kptSIFT_src[best_index][m.queryIdx].pt for m in matches_best]).reshape(-1,1,2)
                                        dst_pts = np.float32([kptSIFT_dst[m.trainIdx].pt for m in matches_best]).reshape(-1,1,2)

                                        # Find Homography Matrix From Corresponding Points In Source And Destination Images
                                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                                        matchesMask = mask.ravel().tolist()

                                        # Set Dimensions To That Of Best Matching Source
                                        h = dims_src[best_index][0]
                                        w = dims_src[best_index][1]

                                        # NumPy Array Of Corner Points Of Source
                                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                                        #If Homography Matrix Is Found
                                        if M is not None:
                                                # Find Points Corresponding To Corners In Source Image
                                                dst = cv2.perspectiveTransform(pts,M)

                                                # Find Bounding Box Coordinates From NumPy Array Of Perspective Corners
                                                [t_xa, t_ya, t_xb, t_yb] = [min(i[0][0] for i in dst), min(i[0][1] for i in dst), max(i[0][0] for i in dst), max(i[0][1] for i in dst)]
                                                #cv2.rectangle(frame, (t_xa, t_ya), (t_xb, t_yb), (255, 255, 255), 2)
                                                cv2.polylines(frame,[np.int32(dst)],True,(255,0,0),2, cv2.LINE_AA)

                                                # Compute Overlap Area Of YOLO Detection With SIFT/FLANN Detection
                                                ovlp_area = (sorted([d_xa, t_xb, d_xb])[1] - sorted([d_xa,t_xa,d_xb])[1])*(sorted([d_ya,t_yb, d_yb])[1] - sorted([d_ya,t_ya,d_yb])[1])

                                                # Compute YOLO Detection Area
                                                dtcn_area = (d_xb - d_xa) * (d_yb - d_ya)

                                                # Verify If SIFT And FLANN Detection Is Actually A Positive
                                                if ovlp_area >= 0.5*dtcn_area :
                                                        # Re-initialize The Tracker On The Current Positive
                                                        trkCSRT = cv2.TrackerCSRT_create()
                                                        trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))   
                                

                                                
        # Update The FPS Counter
        fps.update()
        fps.stop()
        
        # Display FPS On The Frame
        cv2.putText(frame, "{:.2f}".format(fps.fps()), (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display The Frame
        cv2.imshow("Frame", frame)

        # Detect Keypress
        key = cv2.waitKey(1) & 0xFF

        # On Pressing 'q', Break From Infinite Loop
        if key == ord("q"):
                break
        # On Pressing 'c', Cancel The Current Tracking
        elif key == ord("c"):
                init = False


# Release The WebCam Pointer
vs.stop()
 
# Close All Windows
cv2.destroyAllWindows()     






