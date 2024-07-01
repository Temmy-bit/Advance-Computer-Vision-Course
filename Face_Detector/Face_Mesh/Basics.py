import cv2
import mediapipe as mp
import time
from img_source import image_source

# val = q
# cap = cv2.VideoCapture("Building and deploying your first machine learning app in Python using Gradio-(480p).mp4")
cap = image_source()
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(color=(0,222,225),thickness = 2 ,circle_radius=2)


while True:
    success, img = cap.read()
    imageRBG = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imageRBG)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,
                                  mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                x,y = int(lm.x*w),int(lm.y*h)
                print(id,x,y)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,223,255),5)
    cv2.imshow("Image",img)
    cv2.waitKey(1)