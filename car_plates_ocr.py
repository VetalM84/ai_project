"""Detect car plates on images using OpenCV by contours.
Recognize the plate's number using EasyOCR."""

import cv2
import pylab as pl
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


# read the image
img = cv2.imread("media/car_2.jpg")
# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# remove noise
img_filter = cv2.bilateralFilter(img_gray, 9, 15, 15)

edges = cv2.Canny(img_filter, 30, 200)

plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
pl.show()

contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
print(location)

mask = np.zeros(img_gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
bitwise_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(bitwise_image, cv2.COLOR_BGR2RGB))
pl.show()

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = img_gray[x1:x2, y1:y2]

reader = easyocr.Reader(["en"])
result = reader.readtext(cropped_image)
print(result)

text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(
    img,
    text=text,
    org=(approx[0][0][0], approx[1][0][1] + 60),
    fontFace=font,
    fontScale=1,
    color=(0, 255, 0),
    thickness=5,
)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
pl.show()
