import cv2
import time
import mediapipe as mp


class FaceDetector():
    def __init__(self,mindectectionCon=0.5):
          
        self.mindectectionCon = mindectectionCon
          
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.mindectectionCon)

    def findFace(self,img,draw=True):
         
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []

        if self.results.detections:
                # for facelms in results.
                
                for id, detection in enumerate(self.results.detections):
                    # print(detection.location_data.relative_bounding_box)
                    bboxC = detection.location_data.relative_bounding_box
                    h,w,c = img.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h),\
                            int(bboxC.width *w), int(bboxC.height * h)
    
                    bboxs.append([id,bbox,detection.score])

                    if draw:
                        img = self.fancyDraw(img,bbox)
                        cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),
                            cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
                    
        return img,bboxs
 # mpDraw.draw_detection(img,detection)
                    # print(id, detection)
                    # print(detection.score)
                    # p
    def fancyDraw(self,img,bbox,l=30,t= 5,rt = 1):
        x,y,w,h = bbox
        x1, y1 = x+w , y+h

        cv2.rectangle(img,bbox,(255,0,0),rt)
        # for Top left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,0),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,0),t)

        # for Top right x,y
        cv2.line(img,(x1,y),(x1-l,y),(255,0,0),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,0),t)

        # for Bottom left x,y1
        cv2.line(img,(x,y1),(x+l,y1),(255,0,0),t)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,0),t)

        # for Bottom right x,y1
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,0),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,0),t)


        return img
    

def main():
    cap = cv2.VideoCapture(1)
    pTime = 0
    detector = FaceDetector()
    while True:
        
        success, img = cap.read() 
        img, bboxs = detector.findFace(img,True)
        if len(bboxs) != 0:
            print(bboxs)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img,f'FPS: {int(fps)}',(20,90),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(225,2,255),3)
        cv2.imshow('Image',img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()