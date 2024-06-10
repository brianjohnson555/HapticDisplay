import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial

from numpy import genfromtxt

###### USER SETTINGS ######
FILENAME = 'datakoi_output.txt'
COM1 = 'COM9'
COM2 = 'COM10'


data=genfromtxt(FILENAME,delimiter=',')

ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS),
       serial.Serial(COM2, 9600, timeout=0, bytesize=serial.EIGHTBITS)]

def make_packet(periods):
    packetlist = []
    packetlist.append(('P').encode()) # encode start of period array
    for period in periods:
        packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
    packet = b''.join(packetlist) #
    return packet


ser[0].write('E'.encode()) # Enable HV
ser[1].write('E'.encode()) # Enable HV
time.sleep(0.25)

while True:
    periods = []
    packets = []

    output = data # Need to read each line of data, and loop when it runs out

    output[output==0] = 0.01
    output_rec = np.reciprocal(output)
    output_scale = (output_rec*100)
    output_new = output_scale.astype(int)
    output_new[output_new>500] = 0


    periods.append(np.concatenate([output_new[0,0:5], output_new[1,0:5]]))
    periods.append(np.concatenate([output_new[3,0:5], output_new[2,0:5]]))
    packets.append(make_packet(periods[0]))
    packets.append(make_packet(periods[1]))
    ser[0].write(packets[0])
    ser[1].write(packets[1])
    time.sleep(0.05)
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break
    

ser[0].write('D'.encode()) # disable HV  
ser[1].write('D'.encode()) # disable HV  
time.sleep(0.5)  
# ser.close()
