import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial

ser = serial.Serial('COM9', 9600, timeout=0)
ser.bytesize = serial.EIGHTBITS

def make_packets(duties, periods):
    duties_abs = np.int32(np.floor(np.multiply(periods,duties))) # convert from % to msec
    packetlist = []
    packetlist.append(('P').encode()) # encode start of period array
    for duty in duties_abs:
        packetlist.append((duty.item()).to_bytes(2, byteorder='little')) # convert to 16bit
    packet_duty = b''.join(packetlist) # combine packetlist as bytes

    packetlist = []
    packetlist.append(('T').encode()) # encode start of period array
    for period in periods:
        packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit, factor of 2 for period in MCU
    packet_period = b''.join(packetlist) # combine packetlist as bytes

    return packet_duty, packet_period

ser.write('E'.encode()) # Enable HV
time.sleep(0.05)

period = 10*2
duty = 0.05
periods = np.array([period, period, period, period, period, period, period, period, period, period])
duties = np.array([duty, duty, duty, duty, duty, duty, duty, duty, duty, duty])
packet_duty, packet_period = make_packets(duties, periods)
ser.write(packet_duty)
ser.write(packet_period)
time.sleep(3)

period = 10*2
duty = 0.2
periods = np.array([period, period, period, period, period, period, period, period, period, period])
duties = np.array([duty, duty, duty, duty, duty, duty, duty, duty, duty, duty])
packet_duty, packet_period = make_packets(duties, periods)
ser.write(packet_duty)
ser.write(packet_period)
time.sleep(3)
    
period = 10*2
duty = 0.5
periods = np.array([period, period, period, period, period, period, period, period, period, period])
duties = np.array([duty, duty, duty, duty, duty, duty, duty, duty, duty, duty])
packet_duty, packet_period = make_packets(duties, periods)
ser.write(packet_duty)
ser.write(packet_period)
time.sleep(3)

ser.write('D'.encode()) # disable HV  
time.sleep(0.5)  
# ser.close()
