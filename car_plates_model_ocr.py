"""Detect car plates on images using OpenCV and pretrained model.
Recognize the plate's number using EasyOCR."""


import cv2
import easyocr


# load plate recognition model
plates = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
# read the image
img = cv2.imread("media/car_2.jpg")
# convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# remove noise
img_filter = cv2.bilateralFilter(img_gray, 9, 15, 15)
# detect the plates coordinates
plates_rects = plates.detectMultiScale(img_filter, scaleFactor=1.2, minNeighbors=5)

# Define the region of interest as the desired bounding box
try:
    region_x = plates_rects[0][0]
    region_y = plates_rects[0][1]
    region_width = plates_rects[0][2]
    region_height = plates_rects[0][3]
    # Extract the desired image region
    cropped_img = cv2.getRectSubPix(
        img_filter,
        (region_width, region_height),
        (region_x + region_width / 2, region_y + region_height / 2),
    )
    reader = easyocr.Reader(["en"])
    result = reader.readtext(cropped_img)
    print(result)

    cv2.imshow("test", cropped_img)
    cv2.waitKey(0)
except IndexError:
    print("No plates found")
