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
import haptic_utils.haptic_map as haptic_map
import haptic_utils.generator as generator
import haptic_utils.USB as USB

###### MAIN ######

# Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)
time.sleep(0.5)

# Enable HV!!!
serial_writer.HV_enable()

# prepare preprogrammed sequence:
frame_rate = 24

output_data = haptic_map.make_output_data(generator.ramp(direction=1),
                                        freq_range=(0,200),
                                        duty_range=(0.05,0.5))

output_data1 = haptic_map.make_output_data(generator.ramp(direction=-1),
                                                           freq_range=(0,200),
                                                           duty_range=(0.05,0.5))
output_data.extend(output_data1)

output_data1 = haptic_map.make_output_data(generator.sine_global(freq=1),
                                                           freq_range=(0,50),
                                                           duty_range=(0.05,0.5))
output_data.extend(output_data1)

output_data1 = haptic_map.make_output_data(generator.checker_sine(total_time=5,freq=0.5),
                                                           freq_range=(0,20),
                                                           duty_range=(0.5,0.5))
output_data.extend(output_data1)

output_data1 = haptic_map.make_output_data(generator.checker_sine(total_time=5,freq=0.5),
                                                           freq_range=(20,20),
                                                           duty_range=(0.05,0.6))
output_data.extend(output_data1)

while output_data.length()>1:
    t_start = time.time()
    # get latest USB packet and intensity:
    intensity_array, packet_list = output_data.pop()
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
