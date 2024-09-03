#!/usr/bin/env python3

"""This demo script shows the variety of haptic sensations that can be created by the haptic display.

It is a preprogrammed input sequence across a range of frequencies and duty cycles, as well as some
signal patterns (checkerboard, sine wave, etc)."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
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

# Enable HV!!!
USB_writer.HV_enable()

# initialize camera and gesture model:
frame_rate = 24
generator = algo_functions.IntensityGenerator(total_time=5, frame_rate=frame_rate)
intensity_map = algo_functions.IntensityMap()

output_list = []
output_list.extend(generator.ramp(1))
output_list.extend(generator.ramp(-1))
output_list.extend(generator.sine_global(freq=1))
output_list.extend(generator.checker_sine(freq=0.5))
output_list.extend(generator.checker_square(freq=0.5))

while len(output_list)>1:
    t_start = time.time()

    intensity_array = output_list.pop(0)
    haptic_output = intensity_map.map_0_24Hz(intensity_array) # map from algo intensity to duty cycle/period
    USB_writer.write_to_USB(haptic_output)

    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Video', 4*192, 4*108)
    cv2.imshow('Video',intensity_array)

    t_end=time.time()
    t_elapsed = t_end-t_start

    if t_elapsed<1/frame_rate:
        time.sleep(1/frame_rate-(t_elapsed)) #maintain constant loop frame rate (24 fps)

    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!
    
# Disable HV!!!
USB_writer.HV_disable()
time.sleep(0.5)
