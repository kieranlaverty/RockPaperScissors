import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision


TF_ENABLE_ONEDNN_OPTS=0

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.IMAGE)



with GestureRecognizer.create_from_options(options) as recognizer:


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break