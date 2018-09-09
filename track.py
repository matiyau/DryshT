from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import darknet as dn
from copy import deepcopy as dc

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
trkTLD = cv2.TrackerTLD_create()

trkCSRT_cp = cv2.TrackerCSRT_create()
trkTLD_cp = cv2.TrackerTLD_create()
 
# initialize the FPS throughput estimator
fps = FPS().start()

dn_dir = "/home/n7/darknet/"
net = dn.load_net(dn_dir + "cfg/yolov3.cfg", dn_dir + "yolov3.weights", 0)
meta = dn.load_meta(dn_dir + "cfg/coco.data")

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

        if not init :
                r = dn.detect(net, meta, "f.png")        
                for i in r :
                        #print(i[0])
                        if i[0] == "cell phone" :
                                init = True

                                [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                                [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]
                                                                
                                trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))
                                trkTLD.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))
                                break
                                
        trkCSRT_cp = trkCSRT
        trkTLD_cp = trkTLD
        [resCSRT, roiCSRT] = trkCSRT_cp.update(frame)
        [resTLD, roiTLD] = trkTLD_cp.update(frame)

        dtcts = []

        cv2.imwrite("f.png",frame)
        r = dn.detect(net, meta, "f.png")
        
        for i in r :
                #print(i[0])
                if i[0] == "cell phone" :
                        dtcts.append(i)
                        
                        [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                        [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]
                        
                        cv2.rectangle(frame, (d_xa, d_ya), (d_xb, d_yb), (0, 255, 0), 2)
                        
                        
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
                                if resTLD :
                                        [t_xa, t_ya, t_w, t_h] = [int(a) for a in roiTLD]
                                        #cv2.rectangle(frame, (t_xa, t_ya), (t_xa + t_w, t_ya + t_h), (255, 0, 0), 2)
                                        ovlp_area = (sorted([d_xa, t_xa + t_w, d_xb])[1] - sorted([d_xa,t_xa,d_xb])[1])*(sorted([d_ya,t_ya + t_h, d_yb])[1] - sorted([d_ya,t_ya,d_yb])[1])
                                        dtcn_area = (d_xb - d_xa) * (d_yb - d_ya)
                                        #print(ovlp_area,dtcn_area)
                                        if ovlp_area >= 0.5*dtcn_area :
                                                cv2.rectangle(frame, (t_xa, t_ya), (t_xa + t_w, t_ya + t_h), (255, 0, 0), 2)
                                                trkTLD = trkTLD_cp                                
                                        break

                        if dtcts.index(i) == len(dtcts)-1 :
                                resCSRT = False
    

        if (not resCSRT) and resTLD :                        
                for i in dtcts :
                        [d_xa, d_ya, d_xb, d_yb] = [int(i[2][0] - i[2][2]/2), int(i[2][1] - i[2][3]/2), int(i[2][0] + i[2][2]/2), int(i[2][1] + i[2][3]/2)]
                        [d_xa, d_ya, d_xb, d_yb] = [max(0, d_xa), max(0,d_ya), min(d_xb, W), min(d_yb, H)]

                        [t_xa, t_ya, t_w, t_h] = [int(a) for a in roiTLD]

                        cv2.rectangle(frame, (t_xa, t_ya), (t_xa + t_w, t_ya + t_h), (255, 0, 0), 2)
                        
                        ovlp_area = (sorted([d_xa, t_xa + t_w, d_xb])[1] - sorted([d_xa,t_xa,d_xb])[1])*(sorted([d_ya,t_ya + t_h, d_yb])[1] - sorted([d_ya,t_ya,d_yb])[1])
                        dtcn_area = (d_xb - d_xa) * (d_yb - d_ya)
                        if ovlp_area >= 0.5*dtcn_area :
                                #cv2.rectangle(frame, (t_xa, t_ya), (t_xa + t_w, t_ya + t_h), (255, 0, 0), 2)
                                trkCSRT = cv2.TrackerCSRT_create()
                                trkCSRT.init(frame, (d_xa,d_ya,d_xb-d_xa,d_yb-d_ya))
                                trkTLD = trkTLD_cp
                                

                                                
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






