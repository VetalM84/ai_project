"""Detect objects on a video using ImageAI and pyTorch."""

from imageai.Detection import VideoObjectDetection


detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("tiny-yolov3.pt")
detector.loadModel()
detector.detectObjectsFromVideo(
    input_file_path="media/input_imageai.mp4",
    output_file_path="media/output_imageai",
    frames_per_second=20,
    log_progress=True,
)
