###### USER SETTINGS ######
FILENAME = "datakoi_output.txt"
VIDEONAME = "video_koi.mp4"
COM_A = "COM9"
COM_B = "COM15"
COM_C = "COM16"
SERIAL_ACTIVE = True

###### INITIALIZATIONS ######
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial
import algo_functions # my custom file, must be in the same directory
from numpy import genfromtxt

###### MAIN ######
# Load data
data=genfromtxt(FILENAME,delimiter=',')[:,0:28]
data_length = data.shape[0]

# Set up serial list
if SERIAL_ACTIVE:
    ser = [serial.Serial(COM_A, 9600, timeout=0, bytesize=serial.EIGHTBITS), 
           serial.Serial(COM_B, 9600, timeout=0, bytesize=serial.EIGHTBITS),
           serial.Serial(COM_C, 9600, timeout=0, bytesize=serial.EIGHTBITS)]
    
    # ENABLE HV!
    algo_functions.HV_enable(ser)

while True:
    isClose=False # break condition
    cap = cv2.VideoCapture(VIDEONAME) # set up video capture
    dataindex = 0 # initialize data start point

    while True:
        ret,img = cap.read() # read frame from video

        if ret and dataindex<data_length: # if frame exists, run; otherwise, video is finished->loop back to beginning
            intensity_array = data[dataindex,:] # Read each line of data
            dataindex += 1
            duty_array, period_array = algo_functions.map_intensity(intensity_array) # map from algo intensity to duty cycle/period

            if SERIAL_ACTIVE:
                # pack and write data to HV switches:
                algo_functions.packet_and_write(ser, duty_array, period_array)

            # Display video:
            cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video',cv2.flip(img, -1))
            # time.sleep(0.05)

            if(cv2.waitKey(10) & 0xFF == ord('b')): # if user pressed 'b' for break
                isClose = True # assign stop flag
                break
        else: 
            break # video is finished, break and reset frame count

    if isClose: 
        break # user pressed 'b', stop script
    
if SERIAL_ACTIVE:
    # DISABLE HV!
    algo_functions.HV_disable(ser)
    time.sleep(0.5)
