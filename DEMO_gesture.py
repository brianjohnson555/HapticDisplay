import numpy as np
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from matplotlib import pyplot as plt
import time

model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
class callback:
    stored_result = None
    def get_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        if result.gestures:
            self.stored_result = str(result.gestures[0][0].category_name)
        else:
            self.stored_result = 'Null'

s = callback()

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=s.get_result)

cap = cv2.VideoCapture(0) #stream from webcam

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret,img = cap.read()
        frame_timestamp_ms = int(np.floor(time.time() * 1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        
        img_anno = cv2.putText(img, 
                               s.stored_result, 
                               org=(20, 100), 
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                               fontScale=2, 
                               color=(255,0,0), 
                               thickness=2)
        cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Video', 4*192, 4*108)
        cv2.imshow('Video',img_anno)
        if(cv2.waitKey(10) & 0xFF == ord('b')):
            break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!
