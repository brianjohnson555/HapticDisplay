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

# Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
USB_writer = algo_functions.USBWriter(serial_ports, serial_active=SERIAL_ACTIVE)

# Enable HV!!!
USB_writer.HV_enable()

# initialize camera and gesture model:
cap = cv2.VideoCapture(0) #stream from webcam
gesture = algo_gesture.Gesture() # keep track of each recurrance of gestures
recognizer = algo_gesture.Recognizer(gesture)

with recognizer.recognizer as gesture_recognizer: #GestureRecognizer type needs "with...as" in order to run properly (enter()/exit())
    while True:
        ret, frame = cap.read()
        frame_timestamp_ms = int(np.floor(time.time() * 1000))
        gesture.get_latest_gesture(gesture_recognizer, frame_timestamp_ms, frame)
        gesture.update()
        intensity_array = gesture.output_latest
        haptic_output = algo_functions.map_intensity(intensity_array) # map from algo intensity to duty cycle/period
        USB_writer.write_to_USB(haptic_output)

        frame_annotated = cv2.putText(frame, 
                               str(gesture.gesture_active), 
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


    
# Disable HV!!!
USB_writer.HV_disable()
time.sleep(0.5)
