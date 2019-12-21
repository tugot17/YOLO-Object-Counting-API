import cv2

from object_counting_api import ObjectCountingAPI

options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.5, "gpu": 1.0}
IMG_PATH = "sample_inputs/united_nations.jpg"

img = cv2.imread("sample_inputs/united_nations.jpg")
counter = ObjectCountingAPI(options)

counter.count_objects_on_image(img, targeted_classes=["person"], show=True)