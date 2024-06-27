###### USER SETTINGS ######
# SERIAL_ACTIVE = True
# NUM_SWITCHBOARDS = 2
FILENAME = 'datakoi_output.txt'
COM1 = "COM9" #bottom
COM2 = "COM12" #top

###### INITIALIZATIONS ######
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial
from numpy import genfromtxt

###### MAIN ######
# Set up pixel to switchboard mapping:
# bot: |2|4|6|8|10|
# bot: |1|3|5|7|9|
# top: |2|4|6|8|10|
# top: |1|3|5|7|9|
def map_pixels(output):
    periods_bot = np.array([output[1,0], output[0,0], output[1,1], output[0,1], output[1,2], output[0,2], output[1,3], output[0,3], output[1,4], output[0,4]])
    periods_top = np.array([output[3,0], output[2,0], output[3,1], output[2,1], output[3,2], output[2,2], output[3,3], output[2,3], output[3,4], output[2,4]])
    return periods_bot, periods_top

# Load data
data=genfromtxt(FILENAME,delimiter=',')

# Set up serial list
ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS),
       serial.Serial(COM2, 9600, timeout=0, bytesize=serial.EIGHTBITS)]

def make_packet(periods):
    packetlist = []
    packetlist.append(('P').encode()) # encode start of period array
    for period in periods:
        packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
    packet = b''.join(packetlist) # add space
    return packet


ser[0].write('E'.encode()) # Enable HV
ser[1].write('E'.encode()) # Enable HV
time.sleep(0.25)

while True:
    periods = []
    packets = []

    output = data # Need to read each line of data, and loop when it runs out

    ## LINEAR MAPPING FROM INTENSITY TO FREQUENCY (TO DISPLAY)
    mapped_freq = 20*output # mapped frequency (Hz)
    mapped_freq[mapped_freq==0] = 0.01
    mapped_per = np.reciprocal(mapped_freq) # mapped period (sec)
    mapped_per_ms = 1000*mapped_per # mapped period (ms)
    mapped_per_ms = mapped_per_ms.astype(int)
    mapped_per_ms[mapped_per_ms>500] = 0 # anything above 500 ms = 0 (below 2 Hz = 0)

    periods_bot, periods_top = map_pixels(mapped_per_ms)
    packet_bot = make_packet(periods_bot)
    packet_top = make_packet(periods_top)
    ser[0].write(packet_bot)
    ser[1].write(packet_top)
    time.sleep(0.05)

    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break
    

ser[0].write('D'.encode()) # disable HV  
ser[1].write('D'.encode()) # disable HV  
time.sleep(0.5)  
# ser.close()
