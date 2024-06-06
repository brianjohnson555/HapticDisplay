import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial

ser = serial.Serial('COM9', 9600, timeout=0)
ser.bytesize = serial.EIGHTBITS
# ser.rts = False
ser.dtr = False

def make_packet(periods):
    packetlist = []
    packetlist.append(('P').encode()) # encode start of period array
    for period in periods:
        packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
    packet = b''.join(packetlist) #
    return packet

ser.write('E'.encode()) # Enable HV
time.sleep(0.5)

periods = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0])

for k in range(1,201):
    packet = make_packet(k*periods)
    ser.write(packet)
    time.sleep(0.5)

ser.write('D'.encode()) # disable HV  
time.sleep(0.5)  
# ser.close()
