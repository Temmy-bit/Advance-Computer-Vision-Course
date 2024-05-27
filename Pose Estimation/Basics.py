import cv2
import mediapipe as mp
import time

# val = int(input("\n\nInput 0 for system camera and 1 for external camera: "))

class poseDetector():
    def __init__(self,mode=False,complexity=1,upper_body=True,smooth = True,enable_seg = False,smooth_seg=True,detectionCon=0.5,trackCon=0.5):

        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon 
        self.complexity = complexity
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,
                                     self.smooth,self.enable_seg,self.smooth_seg,self.detectionCon,self.trackCon)
    

    def findPose(self,img,draw=True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS)
                # print(self.results.pose_landmarks)

        return img
    
    def findPosition(self,img,draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(cx,cy)
                if id != 0:
                    lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(223,233,32),cv2.FILLED)

        return lmList

def main():

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
        

if __name__ == "__main__":
    main()