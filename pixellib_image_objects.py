"""Detect objects on the image using TensorFlow and PixelLib."""

import os
from pixellib.instance import instance_segmentation


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
segment_image = instance_segmentation(infer_speed="fast")
segment_image.load_model("models/mask_rcnn_coco.h5")  # file is too big, load it from GitHub
segment_image.segmentImage("media/objects.jpg", show_bboxes=True, output_image_name="media/output_objects.jpg")
