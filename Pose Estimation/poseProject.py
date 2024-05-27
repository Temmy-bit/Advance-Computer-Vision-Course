import time
import cv2
from Basics import poseDetector

pTime = 0
cap = cv2.VideoCapture(1)
detector = poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img,draw=False)
    lmList = detector.findPosition(img,draw=False)
    if lmList != []:
        print(lmList[10:16])
        print(lmList[22:28])
        lis = [11,12,13,14,15,16,23,24,25,26,27,28]
        for i in lis:
            cv2.circle(img,(lmList[i][1],lmList[i][2]),7,(0,0,232),cv2.FILLED)
        # cv2.circle(img,(lmList[22:28][1],lmList[22:28][2]),7,(0,232,0),cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(80,90),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)

    cv2.imshow("Video",img)

    cv2.waitKey(1)
    
