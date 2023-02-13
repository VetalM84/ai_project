"""Detect objects on a video from webcam using ImageAI and pyTorch. Show the video using OpenCV."""

import cv2
from imageai.Detection import ObjectDetection


camera = cv2.VideoCapture(0)

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("tiny-yolov3.pt")
detector.loadModel()

while camera.isOpened():
    ret, img = camera.read()
    annotated_image, predictions = detector.detectObjectsFromImage(
        input_image=img,
        output_type="array",
        display_percentage_probability=False,
        display_object_name=True,
    )
    cv2.imshow("Webcam", annotated_image)
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
        break

camera.release()
cv2.destroyAllWindows()
