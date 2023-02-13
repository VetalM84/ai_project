"""Detect car plates on images using OpenCV and pretrained model."""

import cv2


plates = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
# read the image
img = cv2.imread('media/car_3.jpg')
# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# remove noise
img_filter = cv2.bilateralFilter(img_gray, 9, 15, 15)
# detect the plates
plates_rects = plates.detectMultiScale(img_filter, scaleFactor=1.2, minNeighbors=5)
# draw the rectangle around each plate
for (x, y, w, h) in plates_rects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# display the image
cv2.imshow('Detected', img)
cv2.waitKey(0)
