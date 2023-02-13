"""Detect objects on the images using ImageAI and pyTorch."""

from imageai.Detection import ObjectDetection


detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("tiny-yolov3.pt")
detector.loadModel()
# detect custom objects in the image
custom = detector.CustomObjects(person=True, car=True, dog=True, bicycle=True, bus=True, traffic_light=True)

detector.detectObjectsFromImage(
    custom_objects=custom,
    input_image="media/objects.jpg",
    output_image_path="media/detected-objects.jpg",
    minimum_percentage_probability=30,
)
