from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import darknet as dn
from copy import deepcopy as dc

def goodMatches(dsc_src, dsc_dst) :
        matches = flann.knnMatch(dsc_src,dsc_dst,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        return good

def selectObj(event,x,y,flags,param) :
        if not init :
                if event == cv2.EVENT_LBUTTONUP :
                        global click_x
                        global click_y
                        click_x = x
                        click_y = y
        

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
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

trkCSRT = cv2.TrackerCSRT_create()
trkCSRT_cp = cv2.TrackerCSRT_create()

sift = cv2.xfeatures2d.SIFT_create()
kptSIFT_src = [None, None, None, None]
dscSIFT_src = [None, None, None, None]
dims_src = [None,None,None,None]

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
# initialize the FPS throughput estimator
fps = FPS().start()

dn_dir = "/home/n7/darknet/"
net = dn.load_net(dn_dir + "cfg/yolov3.cfg", dn_dir + "yolov3.weights", 0)
meta = dn.load_meta(dn_dir + "cfg/coco.data")


init = False
cv2.namedWindow('Frame')
cv2.setMouseCallback("Frame", selectObj)

click_x = -1
click_y = -1

while True :
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame

        # check to see if we have reached the end of the stream
        if frame is None:
                break

        (H, W) = frame.shape[:2]

        if not init :
                cv2.imwrite("f.png",frame)
                r = dn.detect(net, meta, "f.png")        
                for i in r :
                        #print(i[0])
                        if i[0] == "cell phone" :
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]

                                cv2.rectangle(frame, (d_xa, d_ya), (d_xb, d_yb), (0, 255, 0), 2)

                                if click_x > d_xa and click_x < d_xb and click_y > d_ya and click_y < d_yb :
                                        init = True
                                        click_x = -1
                                        click_y = -1
                                                           
                                        trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))

                                        roi = cv2.cvtColor(frame[d_ya:d_yb, d_xa:d_xb], cv2.COLOR_BGR2GRAY)
                                        cv2.imshow("0", roi)
                                        kptSIFT_src[0], dscSIFT_src[0] = sift.detectAndCompute(roi, None)
                                        dims_src[0] = roi.shape[:2]
                                        roi = imutils.rotate_bound(roi, 90)
                                        cv2.imshow("90", roi)
                                        kptSIFT_src[1], dscSIFT_src[1] = sift.detectAndCompute(roi, None)
                                        dims_src[1] = roi.shape[:2]
                                        roi = imutils.rotate_bound(roi, 90)
                                        cv2.imshow("180", roi)
                                        kptSIFT_src[2], dscSIFT_src[2] = sift.detectAndCompute(roi, None)
                                        dims_src[2] = roi.shape[:2]
                                        roi = imutils.rotate_bound(roi, 90)
                                        cv2.imshow("270", roi)
                                        kptSIFT_src[3], dscSIFT_src[3] = sift.detectAndCompute(roi, None)
                                        dims_src[3] = roi.shape[:2]

                                        break
                        
        else :
                trkCSRT_cp = trkCSRT
                [resCSRT, roiCSRT] = trkCSRT_cp.update(frame)

                dtcts = []

                cv2.imwrite("f.png",frame)
                r = dn.detect(net, meta, "f.png")
                
                for i in r :
                        #print(i[0])
                        if i[0] == "cell phone" :
                                dtcts.append(i)
                                
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]
                                
                                #cv2.rectangle(frame, (d_xa, d_ya), (d_xb, d_yb), (0, 255, 0), 2)
                                
                                
                if resCSRT :
                        for i in dtcts:
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]
                                
                                [t_xa, t_ya, t_w, t_h] = [int(a) for a in roiCSRT]
                                
                                ovlp_area = (sorted([d_xa, t_xa + t_w, d_xb])[1] - sorted([d_xa,t_xa,d_xb])[1])*(sorted([d_ya,t_ya + t_h, d_yb])[1] - sorted([d_ya,t_ya,d_yb])[1])
                                dtcn_area = (d_xb - d_xa) * (d_yb - d_ya)

                                #Detection Of True Positives
                                if ovlp_area >= 0.5*dtcn_area :
                                        cv2.rectangle(frame, (t_xa, t_ya), (t_xa + t_w, t_ya + t_h), (0, 0, 255), 2)
                                        trkCSRT = cv2.TrackerCSRT_create()
                                        trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))
                                        #print("Centre : " + str(t_xa + t_w/2) + ", " + str(t_ya + t_h/20))
                                        break

                                if dtcts.index(i) == len(dtcts)-1 :
                                        resCSRT = False
            

                if not resCSRT :
                        if len(dtcts) :
                                kptSIFT_dst, dscSIFT_dst = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
                                
                        for i in dtcts :
                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]

                                good = [None,None,None,None]
                                good[0] = goodMatches(dscSIFT_src[0], dscSIFT_dst)
                                good[1] = goodMatches(dscSIFT_src[1], dscSIFT_dst)
                                good[2] = goodMatches(dscSIFT_src[2], dscSIFT_dst)
                                good[3] = goodMatches(dscSIFT_src[3], dscSIFT_dst)

                                best_index = 0
                                for index in range (1,4) :
                                        if len(good[index]) > len(good[best_index]) :
                                                best_index = index

                                matches_best = good[best_index]
                                                
                                if len(matches_best)>10 :
                                        #print(best_index)
                                        src_pts = np.float32([kptSIFT_src[best_index][m.queryIdx].pt for m in matches_best]).reshape(-1,1,2)
                                        dst_pts = np.float32([kptSIFT_dst[m.trainIdx].pt for m in matches_best]).reshape(-1,1,2)
                                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                                        matchesMask = mask.ravel().tolist()
                                        h = dims_src[best_index][0]
                                        w = dims_src[best_index][1]
                                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                                        if M is not None:
                                                dst = cv2.perspectiveTransform(pts,M)
                                                
                                                corPts = [[int(i[0][0]), int(i[0][1])] for i in dst]
                                                [t_xa, t_ya, t_xb, t_yb] = [min(i[0][0] for i in dst), min(i[0][1] for i in dst), max(i[0][0] for i in dst), max(i[0][1] for i in dst)]
                                                #cv2.rectangle(frame, (t_xa, t_ya), (t_xb, t_yb), (0, 255, 0), 2)
                                                cv2.polylines(frame,[np.int32(dst)],True,(255,0,0),2, cv2.LINE_AA)

                                                ovlp_area = (sorted([d_xa, t_xb, d_xb])[1] - sorted([d_xa,t_xa,d_xb])[1])*(sorted([d_ya,t_yb, d_yb])[1] - sorted([d_ya,t_ya,d_yb])[1])
                                                dtcn_area = (d_xb - d_xa) * (d_yb - d_ya)

                                                if ovlp_area >= 0.5*dtcn_area :
                                                        trkCSRT = cv2.TrackerCSRT_create()
                                                        trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))   
                                

                                                
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
        elif key == ord("c"):
                init = False


# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
 
# otherwise, release the file pointer
else:
	vs.release()
 
# close all windows
cv2.destroyAllWindows()     






