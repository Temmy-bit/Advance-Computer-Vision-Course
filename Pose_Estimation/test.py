import cv2
import mediapipe
import numpy as np
from poseModule import poseDetector
# from PIL import Image

cap = cv2.VideoCapture(4-3)
detector = poseDetector()

# def convert_to_inches(x, y, pixel_distance, inch_distance):
#     scale_factor = pixel_distance / inch_distance
#     x_inches = x / scale_factor
#     y_inches = y / scale_factor
#     return x_inches, y_inches

# def convert_to_inches2(x_pixel, y_pixel, scale_factor):
#     x_inches = x_pixel / scale_factor
#     y_inches = y_pixel / scale_factor
#     return x_inches, y_inches

width = 980
height = 720
# def estimated_dpi(img,width=width,height=height):
#     wh = width + height
#     image_size = (img.shape[1] + img.shape[0]) * 0.5
#     estimated_dpi = np.divide(wh,image_size)
#     return estimated_dpi

X = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
# print(len(X))
Y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# print(len(Y))
coff = np.polyfit(X,Y,2)
try:
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(width,height))
        # dpi = estimated_dpi(img)
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        

        if len(lmList) != 0:
            # print(lmList[11],lmList[12])
            x1,y1 = lmList[11][1:]
            print("x1,y1",x1,y1)
            x2,y2 = lmList[12][1:]
            print("x2,y2",x2,y2)


            distance = int(np.sqrt((y2 - y1)**2 + (x2 - x1)**2))
            A, B, C = coff
            distanceCM = A*distance**2 + B*distance + C

            print(distanceCM,distance)

            # cv2.rectangle(img,(x1,y1),(x2 - x1, y2 - x1),(255,0,255),3)t

            # pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # # print(pixel_distance)
            # reference_distance = 54

            # scaling_factor = reference_distance / pixel_distance
            # # print(scaling_factor)
            
            # x,y = convert_to_inches2(x1,y1,dpi)
            # # print("Convert to inchices x1,y1",x,y)

            
            # x,y = convert_to_inches2(x2,y2,dpi)
            # # print("Convert to inchices x2,y2",x,y)
            # # x,y = convert_to_inches2(x1,y1,16)
            # print(x,y)

        # cv2.putText(img,f"DPI: {str(int(dpi))}",(20,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("Keyboard Interupt")