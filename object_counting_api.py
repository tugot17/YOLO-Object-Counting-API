from darkflow.net.build import TFNet
from sort import Sort
from utils import COLORS, intersect, get_output_fps_height_and_width
import cv2
import numpy as np

DETECTION_FRAME_THICKNESS = 1

OBJECTS_ON_FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJECTS_ON_FRAME_COUNTER_FONT_SIZE = 0.5


LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
LINE_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
LINE_COUNTER_FONT_SIZE = 2.0
LINE_COUNTER_POSITION = (20, 45)


class ObjectCountingAPI:

    def __init__(self, options):
        self.options = options
        self.tfnet = TFNet(options)

    def _write_quantities(self, frame, labels_quantities_dic):
        for i, (label, quantity) in enumerate(labels_quantities_dic.items()):
            class_id = [i for i, x in enumerate(labels_quantities_dic.keys()) if x == label][0]
            color = [int(c) for c in COLORS[class_id % len(COLORS)]]

            cv2.putText(
                frame,
                f"{label}: {quantity}",
                (10, (i + 1) * 35),
                OBJECTS_ON_FRAME_COUNTER_FONT,
                OBJECTS_ON_FRAME_COUNTER_FONT_SIZE,
                color,
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

    def _draw_detection_results(self, frame, results, labels_quantities_dic):
        for start_point, end_point, label, confidence in results:
            x1, y1 = start_point

            class_id = [i for i, x in enumerate(labels_quantities_dic.keys()) if x == label][0]

            color = [int(c) for c in COLORS[class_id % len(COLORS)]]

            cv2.rectangle(frame, start_point, end_point, color, DETECTION_FRAME_THICKNESS)

            cv2.putText(frame, label, (x1, y1 - 5), OBJECTS_ON_FRAME_COUNTER_FONT, OBJECTS_ON_FRAME_COUNTER_FONT_SIZE, color, 2)

    def _convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(self, objects):
        labels_quantities_dic = {}
        results = []

        for object in objects:
            x1, y1 = object["topleft"]["x"], object["topleft"]["y"]
            x2, y2 = object["bottomright"]["x"], object["bottomright"]["y"]
            confidence = object["confidence"]
            label = object["label"]

            try:
                labels_quantities_dic[label] += 1
            except KeyError:
                labels_quantities_dic[label] = 1

            start_point = (x1, y1)
            end_point = (x2, y2)

            results.append((start_point, end_point, label, confidence))
        return results, labels_quantities_dic

    def count_objects_on_image(self, frame, targeted_classes=[], output_path="count_people_output.jpg", show=False):
        objects = self.tfnet.return_predict(frame)

        if targeted_classes:
            objects = list(filter(lambda res: res["label"] in targeted_classes, objects))

        results, labels_quantities_dic = self._convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(
            objects)

        self._draw_detection_results(frame, results, labels_quantities_dic)

        self._write_quantities(frame, labels_quantities_dic)

        if show:
            cv2.imshow("frame", frame)
            cv2.waitKey()
            cv2.destroyAllWindows()

        cv2.imwrite(output_path, frame)

        # return frame, objects

    def count_objects_on_video(self, cap, targeted_classes=[], output_path="the_output.avi", show=False):
        ret, frame = cap.read()

        fps, height, width = get_output_fps_height_and_width(cap)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while ret:
            objects = self.tfnet.return_predict(frame)

            if targeted_classes:
                objects = list(filter(lambda res: res["label"] in targeted_classes, objects))

            results, labels_quantities_dic = self._convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(
                objects)

            self._draw_detection_results(frame, results, labels_quantities_dic)

            self._write_quantities(frame, labels_quantities_dic)

            output_movie.write(frame)

            if show:
                cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()

    def count_objects_crossing_the_virtual_line(self, cap, line_begin, line_end, targeted_classes=[],
                                                output_path="the_output.avi", show=False):

        ret, frame = cap.read()

        fps, height, width = get_output_fps_height_and_width(cap)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = Sort()
        memory = {}

        line = [line_begin, line_end]
        counter = 0

        while ret:

            objects = self.tfnet.return_predict(frame)

            if targeted_classes:
                objects = list(filter(lambda res: res["label"] in targeted_classes, objects))

            results, _ = self._convert_detections_into_list_of_tuples_and_count_quantity_of_each_label(
                objects)

            # convert to format required for dets [x1, y1, x2, y2, confidence]
            dets = [[*start_point, *end_point] for (start_point, end_point, label, confidence) in results]

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(100)})
            dets = np.asarray(dets)
            tracks = tracker.update(dets)

            boxes = []
            indexIDs = []
            previous = memory.copy()
            memory = {}

            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                memory[indexIDs[-1]] = boxes[-1]

            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))

                    color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                    cv2.rectangle(frame, (x, y), (w, h), color, DETECTION_FRAME_THICKNESS)

                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                        p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                        cv2.line(frame, p0, p1, color, 3)

                        if intersect(p0, p1, line[0], line[1]):
                            counter += 1

                    text = "{}".format(indexIDs[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    i += 1

            cv2.line(frame, line[0], line[1], LINE_COLOR, LINE_THICKNESS)

            cv2.putText(frame, str(counter), LINE_COUNTER_POSITION, LINE_COUNTER_FONT, LINE_COUNTER_FONT_SIZE,
                        LINE_COLOR, 2)

            output_movie.write(frame)

            if show:
                cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    options = {"model": "cfg/yolov2.cfg", "load": "bin/yolov2.weights", "threshold": 0.5, "gpu": 1.0}

    img = cv2.imread("sample_inputs/united_nations.jpg")

    VIDEO_PATH = "sample_inputs/highway_traffic.mp4"

    cap = cv2.VideoCapture(VIDEO_PATH)

    counter = ObjectCountingAPI(options)

    counter.count_objects_crossing_the_virtual_line(cap, line_begin=(100, 300), line_end=(320, 250), show=True)
    # counter.count_objects_on_image(img, targeted_classes=["person"], show=True)