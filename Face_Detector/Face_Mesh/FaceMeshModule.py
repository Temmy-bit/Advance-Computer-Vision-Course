import cv2
import mediapipe as mp
import time
from img_source import image_source


class FaceMeshDetector():

    def __init__(self,static=False,max_face=2,refine_lm=False,detCon=0.5,trackCon=0.5):
        self.static = static
        self.max_face = max_face
        self.refine_lm = refine_lm
        self.detCon = detCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static,self.max_face,
                                                 self.refine_lm,self.detCon,
                                                 self.trackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0,222,225),thickness = 2 ,circle_radius=2)


    def findFaceMesh(self,img,draw=True):
        self.imageRBG = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imageRBG)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,
                                        self.mpFaceMesh.FACEMESH_TESSELATION,
                                        self.drawSpec,self.drawSpec)
                    
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x,y = int(lm.x*w), int(lm.y*h)
                    face.append([x,y])
                    # labeling image with id
                    # cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(225,2,2),1)
            faces.append(face)

        if len(faces) != 0: 
            print(len(faces))

        return img
    
    def findPosition(self,img,draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    h,w,c = img.shape
                    x,y = int(lm.x*w),int(lm.y*h)
                    lmList.append([id,x,y])
                    ig = lmList[:]
                    if draw : # if id == 0 captures lanmark point 0
                        # cv2.circle(img,(x,y),2,(225,0,7),2)
                        # for displaying the part of image to label with id
                        for i in ig:
                            cv2.putText(img,str(i[0]),(i[1:]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        return lmList
    



def main():
    cap = image_source()
    # success, img = cap.read()
    detector = FaceMeshDetector()

    pTime = 0
    while True:
       
        success, img = cap.read()
        detector.findFaceMesh(img,False)
        lmList = detector.findPosition(img,draw=True)
        if len(lmList) != 0:
            print(lmList[0:5])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime 

        cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,223,255),5)
        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()