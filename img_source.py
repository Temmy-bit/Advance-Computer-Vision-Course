import cv2

# val = 4
# print(help(val))
def image_source(source = input("Enter Source Image \n'0': System Camera\n'1': Extenal Camera\nOr Enter Link 2 Video:\n")):

    # print(source)
    if int(source) == 0:
        cap =  cv2.VideoCapture(0)
        # while True:
        #     success, img = cap.read() 

    elif int(source) == 1:
        cap = cv2.VideoCapture(1) 
        # while True:
        #     success, img = cap.read()  
    elif source:
        cap = cv2.VideoCapture(source)

    else: 
        cap = cv2.VideoCapture(0)

    return cap

def main():
    while True:
        
        cap = image_source(1)
        success, img = cap.read()

        cv2.imshow("Image",img)
        cv2.waitKey(1)
    # return img


if __name__ == "__main__":
    main()
