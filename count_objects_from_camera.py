import cv2
from object_counting_api import ObjectCountingAPI

options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.5, "gpu": 1.0}


cap = cv2.VideoCapture(0)

counter = ObjectCountingAPI(options)

counter.count_objects_on_video(cap, show=True)
