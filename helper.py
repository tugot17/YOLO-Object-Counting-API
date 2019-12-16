import cv2


VIDEO_PATH = "/home/piotr/Downloads/Alibi ALI-IPU3030RV IP Camera Highway Surveillance(1).mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

while ret:

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
