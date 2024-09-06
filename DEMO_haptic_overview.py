#!/usr/bin/env python3

"""This demo script shows the variety of haptic sensations that can be created by the haptic display.

It is a preprogrammed input sequence across a range of frequencies and duty cycles, as well as some
signal patterns (checkerboard, sine wave, etc)."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
COM_A = "COM15" # port for MINI switches 1-10
COM_B = "COM9" # port for MINI switches 11-20
# COM_C = "COM16" # port for MINI swiches 21-28

###### INITIALIZATIONS ######
import cv2
import time
import serial
import numpy as np
import visual_haptic_utils.haptic_funcs as haptic_funcs # my custom file
import visual_haptic_utils.USB_writer as USB_writer # my custom file
import matplotlib.pyplot as plt

## debug:
# import warnings
# warnings.filterwarnings("ignore")

###### MAIN ######

# Set up USBWriter:
serial_ports = [COM_A, COM_B]
USB_writer = USB_writer.USBWriter(serial_ports, serial_active=SERIAL_ACTIVE)
time.sleep(0.5)

# Enable HV!!!
USB_writer.HV_enable()

# initialize camera and gesture model:
frame_rate = 24
generator = haptic_funcs.IntensityGenerator(total_time=5, frame_rate=frame_rate)
haptic_map = haptic_funcs.HapticMap()

output_list = []
output_list.extend(generator.ramp(1))
output_list.extend(generator.ramp(-1))
output_list.extend(generator.sine_global(freq=1))
output_list.extend(generator.checker_sine(freq=0.5))
output_list.extend(generator.checker_square(freq=0.5))

duty_array_list, period_array_list = haptic_map.linear_map(output_list,
                                                           freq_range=(0,200),
                                                           duty_range=(0.05,0.5))

while len(output_list)>1:
    t_start = time.time()
    # get latest intensity
    intensity_array = output_list.pop(0)
    # map intensities to freq, duty cycle
    duty_array = duty_array_list.pop(0)
    period_array = period_array_list.pop(0)
    # send to USB
    USB_writer.write_to_USB(duty_array, period_array)
    # create display video
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Video', 1920, 1080)
    cv2.imshow('Video',intensity_array)
    # get elapsed time
    t_end=time.time()
    t_elapsed = t_end-t_start
    # maintain constant loop frame rate (24 fps)
    if t_elapsed<1/frame_rate:
        time.sleep(1/frame_rate-(t_elapsed)) 

    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!
    
# Disable HV!!!
USB_writer.HV_disable()
time.sleep(0.5)
