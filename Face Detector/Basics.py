import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(1)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection( min_detection_confidence=0.5)
while True:
    success, img = cap.read() 
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        # for facelms in results.
        
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h),\
                    int(bboxC.width *w), int(bboxC.height * h)
            cv2.rectangle(img,bbox,(255,0,0),2)
            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),
                cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,90),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(225,2,255),3)
    cv2.imshow('Image',img)

    cv2.waitKey(1)