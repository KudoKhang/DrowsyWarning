from imutils import face_utils
import numpy as np
import dlib
import cv2
from pygame import mixer

mixer.init()
mixer.music.load("Sources/alert.wav")


# Tinh khoang cach giua 2 mat
def e_dist(A, B):
    return np.linalg.norm(A - B)

def eye_ratio(eye):
    d_V1 = e_dist(eye[1], eye[5])
    d_V2 = e_dist(eye[2], eye[4])
    d_H = e_dist(eye[0], eye[3])
    eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)
    return eye_ratio_val

EYE_RATIO_THRESHOLD=0.25
MAX_SLEEP_FRAMES=16
SLEEP_FRAMES=0
COLOR=(0, 255, 0)

face_detect = cv2.CascadeClassifier("Sources/haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("Sources/shape_predictor_68_face_landmarks.dat")

# Lay danh sach cac cum diem landmark cho 2 mat
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(100,100),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        leftEye = landmark[left_eye_start:left_eye_end] # 6 landmark
        rightEye = landmark[right_eye_start:right_eye_end]
        left_eye_ratio = eye_ratio(leftEye)
        right_eye_ratio = eye_ratio(rightEye)

        eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        left_eye_bound = cv2.convexHull(leftEye)
        right_eye_bound = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [left_eye_bound], -1, COLOR, 1)
        cv2.drawContours(frame, [right_eye_bound], -1, COLOR, 1)

        if eye_avg_ratio < EYE_RATIO_THRESHOLD:
            SLEEP_FRAMES += 1
            
            if SLEEP_FRAMES >= MAX_SLEEP_FRAMES:
                cv2.rectangle(frame, (500, 200), (800, 270), (0, 0, 255), -1)
                cv2.putText(frame, "WARNING!!!", (510, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"EYE RATIO: {round(eye_avg_ratio, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
                mixer.music.play()

        else:
            SLEEP_FRAMES = 0            
            cv2.putText(frame, f"EYE RATIO: {round(eye_avg_ratio, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
 

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()

# Trust me, I'm a programmer