import streamlit as st
import cv2
# from img_source import image_source
from Face_Detector.Face_Mesh import FaceMeshModule
# import gradio as gd
try:
    # cap.set(3,720)
    # cap.set(4,720)
    st.write("My image input app")
    img = st.camera_input("cap")
    # cap = image_source()
    if img:
       img = st.image(img)

        # st.write(img.UploadedFile.upload_url)
       detector = FaceMeshModule.FaceMeshDetector()
       detector.findFaceMesh(img)
    while True:
        # img, success = cap.read()
        img = cv2.resize(img,(720,720))

        cv2.imshow("Image",img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Keyboard Interrupt")

except ValueError:
    print("Input from given options or by link")
    # cap = image_source()
# finally:
#     print("Enter Image Source")
