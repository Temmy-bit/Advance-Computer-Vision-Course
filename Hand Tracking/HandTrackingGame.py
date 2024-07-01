import time
import cv2
from HandTrackingModule import handDetector

cTime = 0
pTime = 0
# try:
cap = cv2.VideoCapture(1)
success, img = cap.read()

detector = handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    # print(lmList[2]) # this print out point position values 
    # if len(lmList) != 0: 
    #     print(lmList[::5])
    #     print(lmList[1])
    #     print(lmList[17])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(223,5,225),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)