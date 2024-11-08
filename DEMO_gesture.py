#!/usr/bin/env python3

"""This demo script runs real-time video streaming with gesture-based haptic feedback.

The script creates a video of the computer webcam view, which should be lined up to 
match the position of the haptic display on the screen. Using a pre-trained model, the
hand gesture of a person in the video is automatically detected. Each unique gesture causes
a corresponding unique, pre-programmed haptic output on the haptic display."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False
 # if False, just runs the algorithm without sending to HV switches
VIEW_INTENSITY = True # if True, opens a video showing the depth/intensity map
SAVE_VIDEO = True # if True, saves a video file of both the camera view and the intensity map
COM_A = "COM9" # port for MINI switches 1-10
COM_B = "COM14" # port for MINI switches 11-20
COM_C = "COM15" # port for MINI swiches 21-28

###### INITIALIZATIONS ######
import cv2
import time
import numpy as np
import haptic_utils.USB as USB # my custom file
import haptic_utils.gesture as gesture # my custom file
import matplotlib.pyplot as plt

## debug:
import warnings
warnings.filterwarnings("ignore")

###### MAIN ######

# Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)
if SAVE_VIDEO:
    outlist_image = []
    outlist_haptic = []

# Enable HV!!!
serial_writer.HV_enable()

# initialize camera and gesture model:
cap = cv2.VideoCapture(0) #stream from webcam
gesturer = gesture.Gesture() # keep track of each recurrance of gestures
recognizer = gesture.Recognizer(gesturer)

with recognizer.recognizer as gesture_recognizer: #GestureRecognizer type needs "with...as" in order to run properly (enter()/exit())
    while True:
        ret, frame = cap.read()
        frame_timestamp_ms = int(np.floor(time.time() * 1000))
        gesturer.get_latest_gesture(gesture_recognizer, frame_timestamp_ms, frame)
        intensity, packets = gesturer.output.pop()
        serial_writer.write_packets_to_USB(packets)

        frame_annotated = cv2.putText(frame, 
                               str(gesturer.gesture_active), 
                               org=(20, 60), 
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                               fontScale=1.5, 
                               color=(0,0,0), 
                               thickness=2) # add current gesture annotation to image
        cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Video', 4*192, 4*108)
        cv2.imshow('Video',frame_annotated)

        if VIEW_INTENSITY:
            cv2.namedWindow('Intensity',cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Intensity', 4*192, 4*108)
            cv2.imshow('Intensity',intensity)

        if SAVE_VIDEO:
            outlist_image.append(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB))
            outlist_haptic.append(intensity)

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
    filename = "output_videos/gesture_output_camera.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=24)
    plt.close()

    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist_haptic)):
        im = plt.imshow(outlist_haptic[i], animated=True, cmap='gist_gray', vmin=0, vmax=1)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "output_videos/gesture_output_intensity.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=24)
    plt.close()
    
# Disable HV!!!
serial_writer.HV_disable()
time.sleep(0.5)
