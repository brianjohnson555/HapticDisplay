###### USER SETTINGS ######
# SERIAL_ACTIVE = True
# NUM_SWITCHBOARDS = 2
FILENAME = 'datakoi_output.txt'
COM1 = "COM9" #bottom
COM2 = "COM13" #top
SERIAL_ACTIVE = True

###### INITIALIZATIONS ######
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial
from numpy import genfromtxt

video = 'koi'
###### MAIN ######
# Set up pixel to switchboard mapping:
# bot: |2|4|6|8|10|
# bot: |1|3|5|7|9|
# top: |2|4|6|8|10|
# top: |1|3|5|7|9|
def map_pixels(output):
    periods_bot = np.array([output[5], output[0], output[6], output[1], output[7], output[2], output[8], output[3], output[9], output[4]])
    periods_top = np.array([output[15], output[10], output[16], output[11], output[17], output[12], output[18], output[13], output[19], output[14]])
    return periods_bot, periods_top

# Load data
data=genfromtxt(FILENAME,delimiter=',')[:,0:20]

# Set up serial list
if SERIAL_ACTIVE:
    ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS), 
           serial.Serial(COM2, 9600, timeout=0, bytesize=serial.EIGHTBITS)]

def make_packet(periods):
    packetlist = []
    packetlist.append(('P').encode()) # encode start of period array
    for period in periods:
        packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
    packet = b''.join(packetlist) # convert list to bytes
    return packet

if SERIAL_ACTIVE:
    ser[0].write('E'.encode()) # Enable HV
    ser[1].write('E'.encode()) # Enable HV
    time.sleep(0.25)


while True:
    isClose=False
    cap = cv2.VideoCapture("video_" + video + ".mp4") #use video
    dataindex = 0
    while True:
        ret,img = cap.read()
        if(ret==True):
            periods = []
            packets = []
            
            output = data[dataindex,:] # Read each line of data
            if dataindex>1798:
                dataindex=0 # loop back to start when you run out of data

            ## LINEAR MAPPING FROM INTENSITY TO FREQUENCY (TO DISPLAY)
            mapped_freq = 24*output # mapped frequency (Hz)
            mapped_freq[mapped_freq==0] = 0.01
            mapped_per = np.reciprocal(mapped_freq) # mapped period (sec)
            mapped_per_ms = 1000*mapped_per # mapped period (ms)
            mapped_per_ms = mapped_per_ms.astype(int)
            mapped_per_ms[mapped_per_ms<50] = 50 # anything below 20 ms = 20
            mapped_per_ms[mapped_per_ms>500] = 0 # anything above 500 ms = 0 (below 2 Hz = 0)


            periods_bot, periods_top = map_pixels(mapped_per_ms)
            packet_bot = make_packet(periods_bot)
            packet_top = make_packet(periods_top)

            if SERIAL_ACTIVE:
                ser[0].write(packet_bot)
                ser[1].write(packet_top)

            cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video',img)

            dataindex += 1

            # time.sleep(0.05)
            if(cv2.waitKey(10) & 0xFF == ord('b')):
                isClose = True
                break
        else:  
            break
    if isClose:
        break
    
if SERIAL_ACTIVE:
    ser[0].write('D'.encode()) # disable HV  
    ser[1].write('D'.encode()) # disable HV  
    time.sleep(0.5)  
    # ser.close()
