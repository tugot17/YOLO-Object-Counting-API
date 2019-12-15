from darkflow.net.build import TFNet
import cv2
from sort import *


options = {"model": "cfg/yolov2-tiny.cfg", "load": "cfg/yolov2-tiny.weights", "threshold" : 0.4, "gpu" : 1.0, "labels": "cfg/labels.txt"}
tfnet = TFNet(options)
COLORS = np.random.randint(0, 255, size=(200, 3),
    dtype="uint8")

tracker = Sort()
memory = {}

line_x1 = 50
line_y1 = 210

line_x2 = 400
line_y2 = 210

line = [(line_x1, line_y1), (line_x2, line_y2)]
counter = 0


VIDEO_PATH = "input/vehicle_survaillance.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 450))

    reslts = tfnet.return_predict(frame)

    reslts = list(filter(lambda res: res["label"] == "Car", reslts))

    dets = []

    for result in reslts:

        x1, y1 = result["topleft"]["x"], result["topleft"]["y"]
        x2, y2 = result["bottomright"]["x"], result["bottomright"]["y"]
        confidence = result["confidence"]

        dets.append([x1, y1, x2, y2, confidence])

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x1)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)

                if intersect(p0, p1, line[0], line[1]):
                    counter += 1

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # draw line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

    # draw counter
    cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
