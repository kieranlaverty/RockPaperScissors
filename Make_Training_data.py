import os

import cv2 as cv
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time

#If this is true under the get annotation from frame function
#save the data point to a csv
save = False

#This is prompting for the type of gesture that is being recorded
gesture = input("What is the gesture\n")

TF_ENABLE_ONEDNN_OPTS=0


#for the time stamp of a frame
frame_timestamp_ms = int(time.time() * 1000)


#options for mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='task/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=.3,
    min_tracking_confidence = .3)
detector = vision.HandLandmarker.create_from_options(options)


#font settings
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

#make the csv
def create_csv(data):

    #enumerate over the values of the hand
    for idx, i in enumerate(data):
        print(f'{idx} {i.x} {i.y} {i.z}')

#get the media pipe data from a frame and then calls a draw function
def get_annotation_from(frame):

    #converts image to a type mediapipe can use
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    #gets results from model
    detection_result = detector.detect(image)

    #gets the annated image back
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    #returns detection results and annotated image
    return detection_result, annotated_image


#draw the landmarker on an image
def draw_landmarks_on_image(rgb_image, detection_result):

    #simplifies the data
    hand_landmarks_list = detection_result.hand_landmarks

    #get list of handedness
    handedness_list = detection_result.handedness

    #creates a copy of the image
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):

        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])


        create_csv(hand_landmarks_proto.landmark)


        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())
        

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
        
    print("End set ___________________________\n")
        
    return annotated_image


#create a opencv session
cap = cv.VideoCapture(0)

#video loop
while True:
    
    #get frame
    ret, frame = cap.read()

    #frame picture to be normal video
    frame = cv.flip(frame, 1)

    #if frame exists
    if ret:
        #gets new image
        detection_result, annotation = get_annotation_from(frame)
    
        cv.imshow('', annotation)  
    else:
        print("! No frame")
        

    #if retrieved is false print error
    if not ret:
        print("error in retrieving frame")
        break

    
    if cv.waitKey(1) == ord('q'):
        break
