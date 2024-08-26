###### USER SETTINGS ######
COM_A = "COM9"
COM_B = "COM15"
COM_C = "COM16"
SERIAL_ACTIVE = False

###### INITIALIZATIONS ######
import cv2
import time
import serial
import numpy as np
import utils.algo_functions as algo_functions # my custom file
import utils.algo_gesture as algo_gesture # my custom file

###### MAIN ######

if SERIAL_ACTIVE:
    ser = [serial.Serial(COM_A, 9600, timeout=0, bytesize=serial.EIGHTBITS), 
           serial.Serial(COM_B, 9600, timeout=0, bytesize=serial.EIGHTBITS),
           serial.Serial(COM_C, 9600, timeout=0, bytesize=serial.EIGHTBITS)]
    # ENABLE HV!
    algo_functions.HV_enable(ser)

# initialize camera and gesture model:
cap = cv2.VideoCapture(0) #stream from webcam
GestureRecognizer, call_back = algo_gesture.initialize_gesture_recognizer()
gesture_count = algo_gesture.GestureCount() # keep track of each recurrance of gestures

with GestureRecognizer as recognizer:
    while True:
        ret, frame = cap.read()
        frame_timestamp_ms = int(np.floor(time.time() * 1000))
        algo_gesture.recognize_gesture(recognizer, frame_timestamp_ms, frame) # run code to recognize gesture in camera frame using livestream method
        intensity_array = algo_gesture.gesture_update_loop(gesture_count, call_back.gesture)
        duty_array, period_array = algo_functions.map_intensity(intensity_array) # map from algo intensity to duty cycle/period

        if SERIAL_ACTIVE:
            # pack and write data to HV switches:
            algo_functions.packet_and_write(ser, duty_array, period_array)

        frame_annotated = cv2.putText(frame, 
                               str(gesture_count.active_gesture), 
                               org=(20, 200), 
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                               fontScale=2, 
                               color=(255,0,0), 
                               thickness=2) # add current gesture annotation to image
        cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Video', 4*192, 4*108)
        cv2.imshow('Video',frame_annotated)

        if(cv2.waitKey(10) & 0xFF == ord('b')):
            break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!


if SERIAL_ACTIVE:
    # DISABLE HV!
    algo_functions.HV_disable(ser)
    time.sleep(0.5)

