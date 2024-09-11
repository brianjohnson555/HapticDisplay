#!/usr/bin/env python3

"""This demo script shows the variety of haptic sensations that can be created by the haptic display.

It is a preprogrammed input sequence across a range of frequencies and duty cycles, as well as some
signal patterns (checkerboard, sine wave, etc)."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
COM_A = "COM15" # port for MINI switches 1-10
COM_B = "COM9" # port for MINI switches 11-20
COM_C = "COM16" # port for MINI swiches 21-28

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
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB_writer.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)
time.sleep(0.5)

# Enable HV!!!
serial_writer.HV_enable()

# prepare preprogrammed sequence:
frame_rate = 24
generator = haptic_funcs.IntensityGenerator(total_time=5, frame_rate=frame_rate)
haptic_map = haptic_funcs.HapticMap()
packet_sequence = []
intensity_sequence = []

duty_array_list, period_array_list = haptic_map.linear_map_sequence(generator.ramp(1),
                                                           freq_range=(0,200),
                                                           duty_range=(0.05,0.5))
packet_sequence.extend(USB_writer.make_packet_sequence(duty_array_list, period_array_list))
intensity_sequence.extend(generator.ramp(1))

duty_array_list, period_array_list = haptic_map.linear_map_sequence(generator.ramp(-1),
                                                           freq_range=(0,200),
                                                           duty_range=(0.05,0.5))
packet_sequence.extend(USB_writer.make_packet_sequence(duty_array_list, period_array_list))
intensity_sequence.extend(generator.ramp(-1))

duty_array_list, period_array_list = haptic_map.linear_map_sequence(generator.sine_global(freq=1),
                                                           freq_range=(0,50),
                                                           duty_range=(0.05,0.5))
packet_sequence.extend(USB_writer.make_packet_sequence(duty_array_list, period_array_list))
intensity_sequence.extend(generator.sine_global(freq=1))

duty_array_list, period_array_list = haptic_map.linear_map_sequence(generator.checker_sine(freq=0.5),
                                                           freq_range=(0,20),
                                                           duty_range=(0.5,0.5))
packet_sequence.extend(USB_writer.make_packet_sequence(duty_array_list, period_array_list))
intensity_sequence.extend(generator.checker_sine(freq=0.5))

duty_array_list, period_array_list = haptic_map.linear_map_sequence(generator.checker_sine(freq=0.5),
                                                           freq_range=(20,20),
                                                           duty_range=(0.05,0.6))
packet_sequence.extend(USB_writer.make_packet_sequence(duty_array_list, period_array_list))
intensity_sequence.extend(generator.checker_sine(freq=0.5))

# reverse lists so that pop() will draw from first item in time (faster pop):
packet_sequence.reverse() 
intensity_sequence.reverse()

while len(packet_sequence)>1:
    t_start = time.time()
    # get latest USB packet and intensity:
    packet_list = packet_sequence.pop()
    intensity_array = intensity_sequence.pop()
    # send to USB:
    serial_writer.write_packets_to_USB(packet_list)
    # create display video:
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Video', 1920, 1080)
    cv2.imshow('Video',intensity_array)
    # get elapsed time:
    t_end=time.time()
    t_elapsed = t_end-t_start
    # maintain constant loop frame rate:
    if t_elapsed<1/frame_rate:
        time.sleep(1/frame_rate-(t_elapsed)) 

    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!
    
# Disable HV!!!
serial_writer.HV_disable()
time.sleep(0.5)
