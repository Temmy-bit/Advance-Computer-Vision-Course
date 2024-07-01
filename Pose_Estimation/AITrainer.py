import cv2
import time
import numpy as np
import poseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

count = 0
dir = 0
dir2 = 0 
count2 = 0


pTime = 0
try:
    while True:
        success, img = cap.read()
        cap.set(cv2.CAP_PROP_FPS,60)
        img = cv2.resize(img,(980,720))
        img = detector.findPose(img,False)
        lmList = detector.findPosition(img,False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle,(40,140),(0,100))
            # print(lmList[2])
            angle2 = detector.findAngle(img, 10, 12, 14)
            per2 = np.interp(angle2,(210,310),(0,100))

            # print(angle,per)
            # print(angle2,per2,"\n")

            # check for the dumbbell curls
            if per == 100:
                if dir == 0:
                    count += 0.5
                    dir = 1

            if per2 == 100:
                if dir2 == 0:
                    count2 += 0.5
                    dir2 = 1

            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

                    
            if per2 == 0:
                if dir2 == 1:
                    count2 += 0.5
                    dir2 = 0

            # print(count)
        cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
        cv2.rectangle(img,(720,920),(1050,450),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{str(int(count))}",(45,670), cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),15)
        cv2.putText(img,f"{str(int(count2))}",(770,700), cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),15)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img,f"FPS: {int(fps)}",(20,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Keyboard Interrupt")

