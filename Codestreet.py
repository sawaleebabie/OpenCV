import cv2
import numpy as np
from matplotlib import pyplot as plt

scal_x=0.5
scal_y=0.5
#Open Video and Picture
video = cv2.VideoCapture("street.mp4")

###############################

while True:

    ret, video2 = video.read()
    frame2 = cv2.resize(video2,None,fx=scal_x,fy=scal_y,interpolation=cv2.INTER_AREA)
    img_run2 = frame2


    blur = cv2.blur(img_run2, (5,5))
    hsv2 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #Find Yellow
    lower_yellow = np.array([5, 50, 50])
    upper_yellow = np.array([15, 255, 255])
    mask_yellow = cv2.inRange(hsv2, lower_yellow, upper_yellow)

    ret, contours_yellow, ret= cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

     #Loop Edge of Yellow color
    for contour in contours_yellow:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(img_run2, contour, -1, (0, 0, 0), 3)
   
    #Show result
    cv2.imshow("Follow edge Image", img_run2)

    #Exit Program
    exit = cv2.waitKey(50)
    if exit == 113:
        break

video.release()
cv2.destroyAllWindows()

