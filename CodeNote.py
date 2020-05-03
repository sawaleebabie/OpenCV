import cv2
import numpy as np


scal_x=0.5
scal_y=0.5
#Open Video and Picture
video = cv2.VideoCapture(0)
picture = cv2.VideoCapture("red.jpg")


###############################
ret, pic = picture.read()
x = 150
y = 153
width = 190
height = 230
Img = pic[y: y + height, x: x + width]
hsv_Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv_Img], [0], None, [100], [0, 190])
hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:

    ret, video1 = video.read()
    #frame1 = cv2.resize(video1,None,fx=scal_x,fy=scal_y,interpolation=cv2.INTER_AREA)
    img_run1 = video1 #frame1

    ret, video2 = video.read()
    #frame2 = cv2.resize(video2,None,fx=scal_x,fy=scal_y,interpolation=cv2.INTER_AREA)
    img_run2 = video2 #frame2

    hsv = cv2.cvtColor(img_run1, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    ret, track_window = cv2.meanShift(mask, (x, y, width, height), criteria)
    x, y, w, h = track_window
    cv2.rectangle(img_run1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    blur = cv2.blur(img_run2, (5,5))
    hsv2 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #Find Yellow
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv2, lower_yellow, upper_yellow)
    #Find Red
    lower_red = np.array([160, 20, 70])
    upper_red = np.array([190, 255, 255])
    mask_red = cv2.inRange(hsv2, lower_red, upper_red)
    #Find Green
    lower_green = np.array([24, 131, 93])
    upper_green = np.array([484, 253, 205])
    mask_green = cv2.inRange(hsv2, lower_green, upper_green)

    #Edge Detection Image
    edges = cv2.Canny(img_run2, 100, 200)

    contours_yellow, hierarchy = cv2.findContours(mask_yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_red, hierarchy = cv2.findContours(mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_green, hierarchy = cv2.findContours(mask_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

     #Loop Edge of Yellow color
    for contour in contours_yellow:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(img_run2, contour, -1, (0, 0, 0), 3)
    #Edge of Red color
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(img_run2, contour, -1, (0, 0, 255), 3)
    #Edge of Green color
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 100:
           cv2.drawContours(img_run2, contour, -1, (255, 0, 0), 3)

    #Show result
    cv2.imshow("Edges Video", edges)
    cv2.imshow("Follow edge Image", img_run2)
    cv2.imshow("Edge detect Image", img_run1)

    #Exit Program
    exit = cv2.waitKey(50)
    if exit == 113:
        break

video.release()
cv2.destroyAllWindows()

