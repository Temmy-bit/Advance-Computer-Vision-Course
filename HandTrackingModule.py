import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False, maxHands=2,complex = 1, detectionCon=0.2,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complex = complex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complex,
                                        self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw=True):
        imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:

                if draw == True:
                    self.mpDraw.draw_landmarks(img,handlms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
                    
    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id,lm in enumerate(handlms.landmark):
                    print(id,lm) # id = landmark id, lm = x,y,z
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                    print(f'ID: {id},width: {cx},height: {cy}')
                    if draw : # if id == 0 captures lanmark point 0
                        cv2.circle(img,(cx,cy),2,(225,0,7),cv2.FILLED)
        return lmList

def main():
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
        # print(lmList[2])
        if len(lmList) != 0: 
            print(lmList[::5])
            print(lmList[1])
            print(lmList[17])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(223,5,225),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()