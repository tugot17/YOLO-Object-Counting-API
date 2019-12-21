import cv2
from object_counting_api import ObjectCountingAPI

options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.5, "gpu": 1.0}
VIDEO_PATH = "sample_inputs/highway_traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
counter = ObjectCountingAPI(options)

counter.count_objects_crossing_the_virtual_line(cap, line_begin=(100, 300), line_end=(320, 250), show=True)
