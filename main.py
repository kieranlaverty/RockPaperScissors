import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


TF_ENABLE_ONEDNN_OPTS=0

frame_timestamp_ms = int(time.time() * 1000)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='task/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)


#create a opencv session
cap = cv.VideoCapture(0)

#video loop
while True:
    
    #get frame
    ret, frame = cap.read()

    #frame picture to be normal video
    frame = cv.flip(frame, 1)

    with HandLandmarker.create_from_options(options) as landmarker:
        #turn the frame into an mediapipe frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        #is their a hand
        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        print(hand_landmarker_result)

    #show the frame
    cv.imshow('frame',frame)

    #if retrieved is false print error
    if not ret:
        print("error in retrieving frame")
        break

    
    if cv.waitKey(1) == ord('q'):
        break





