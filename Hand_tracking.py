import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0
while True:
    success, img = cap.read()

    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:

            for id,lm in enumerate(handlms.landmark):
                print(id,lm) # id = landmark id, lm = x,y,z
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(f'ID: {id},width: {cx},height: {cy}')
                if id == 0 : # if id == 0 captures lanmark point 0
                    cv2.circle(img,(cx,cy),15,(225,0,7),cv2.FILLED)


            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(223,5,225),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)




