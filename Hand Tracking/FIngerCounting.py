import cv2
import mediapipe as mp
import os
import time
from HandTrackingModule import handDetector 

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = 'images'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []


for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    # print(f"{folderPath}/{imPath}")

    overlayList.append(image)

print(len(overlayList))
pTime = 0 

detector = handDetector(detectionCon=0.5)

tipIds = [4, 8, 12, 16, 20]

try:
    while True:

        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False)
        fingers = []
        if len(lmList) != 0:
            # print(lmList)
            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)

            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1,5):

                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)

                else:
                    fingers.append(0)

            # print(fingers)
                    
            totalFingers = fingers.count(1)
            print("Total Finger's:",totalFingers)


            h,w,c = overlayList[totalFingers-1].shape
            img[0:h,0:w] = overlayList[totalFingers-1]

            cv2.rectangle(img,(20,225),(170,425),(0,225,0),cv2.FILLED)
            cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(225,0,0),25)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,f"FPS: {int(fps)}",(370,70),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),3)
        
        cv2.imshow("Image",img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Keyboard Interrupt")
