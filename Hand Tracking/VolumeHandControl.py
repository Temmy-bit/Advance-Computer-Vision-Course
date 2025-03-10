import cv2
import mediapipe as mp 
from HandTrackingModule import handDetector
import time
import numpy as np
import math
# from img_source import image_source
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


device = AudioUtilities.GetSpeakers()
interface = device.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL,None)
 
volume = cast(interface,POINTER(IAudioEndpointVolume))
VolRange = volume.GetVolumeRange()
print(VolRange)
volume.SetMasterVolumeLevel(0,None)

minVol = VolRange[0]
maxVol = VolRange[1]

###############################
wCam, hCam = 640,480
###############################


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0 

detector = handDetector(detectionCon=0.5)

vol = 0
volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=True)
    if len(lmList) != 0:
        # print(lmList[4],lmList[8])

        x1, y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1 + x2)// 2, (y1+y2) // 2

        cv2.circle(img, (x1,y1),10,(255,0,255),cv2.FILLED) 
        cv2.circle(img, (x2,y2),10,(255,0,255),cv2.FILLED) 
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img, (cx,cy),10,(255,0,255),cv2.FILLED)

        length = math.hypot(x2 - x1, y2 -y1)
        # print(length)

        # Hand Range 50 - 300
        # volume Range -95.25 - 0

        vol = np.interp(length,[50,300],[minVol,maxVol])
        volBar = np.interp(length,[50,300],[400,150])
        volPer = np.interp(length,[50,300],[0,100])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol,None)

        if length < 50:
            cv2.circle(img, (cx,cy),10,(0,255,0),cv2.FILLED)

    cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
    cv2.putText(img,f'{int(volPer)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(225,0,0),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)
