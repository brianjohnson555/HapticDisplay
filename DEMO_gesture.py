#!/usr/bin/env python3

"""This demo script runs real-time video streaming with gesture-based haptic feedback.

The script creates a video of the computer webcam view, which should be lined up to 
match the position of the haptic display on the screen. Using a pre-trained model, the
hand gesture of a person in the video is automatically detected. Each unique gesture causes
a corresponding unique, pre-programmed haptic output on the haptic display."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
VIEW_INTENSITY = True # if True, opens a video showing the depth/intensity map
SAVE_VIDEO = False # if True, saves a video file of both the camera view and the intensity map
COM_A = "COM9" # port for MINI switches 1-10
COM_B = "COM15" # port for MINI switches 11-20
COM_C = "COM16" # port for MINI swiches 21-28

###### INITIALIZATIONS ######
import cv2
import time
import serial
import numpy as np
import utils.algo_functions as algo_functions # my custom file
import utils.algo_gesture as algo_gesture # my custom file
import matplotlib.pyplot as plt

## debug:
# import warnings
# warnings.filterwarnings("ignore")

###### MAIN ######

# Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
USB_writer = algo_functions.USBWriter(serial_ports, serial_active=SERIAL_ACTIVE)
if SAVE_VIDEO:
    outlist_image = []
    outlist_haptic = []

# Enable HV!!!
USB_writer.HV_enable()

# initialize camera and gesture model:
cap = cv2.VideoCapture(0) #stream from webcam
gesture = algo_gesture.Gesture() # keep track of each recurrance of gestures
recognizer = algo_gesture.Recognizer(gesture)
intensity_map = algo_functions.IntensityMap()

with recognizer.recognizer as gesture_recognizer: #GestureRecognizer type needs "with...as" in order to run properly (enter()/exit())
    while True:
        ret, frame = cap.read()
        frame_timestamp_ms = int(np.floor(time.time() * 1000))
        gesture.get_latest_gesture(gesture_recognizer, frame_timestamp_ms, frame)

        intensity_array = gesture.output_latest
        haptic_output = intensity_map.map_0_24Hz(intensity_array) # map from algo intensity to duty cycle/period
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

        if VIEW_INTENSITY:
            cv2.namedWindow('Intensity',cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Intensity', 4*192, 4*108)
            cv2.imshow('Intensity',intensity_array)

        if SAVE_VIDEO:
            outlist_image.append(frame_annotated)
            outlist_haptic.append(intensity_array)

        if(cv2.waitKey(10) & 0xFF == ord('b')):
            break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!

if SAVE_VIDEO:
    import matplotlib.animation as animation
    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist_image)):
        im = plt.imshow(outlist_image[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "OutputVideos/gesture_output_camera.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=24)
    plt.close()

    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist_haptic)):
        im = plt.imshow(outlist_haptic[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "OutputVideos/gesture_output_intensity.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=24)
    plt.close()
    
# Disable HV!!!
USB_writer.HV_disable()
time.sleep(0.5)
