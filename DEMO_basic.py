#!/usr/bin/env python3

"""This demo script shows a basic single output sequence from the haptic display."""

###### USER SETTINGS ######
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
COM_A = "COM9" # port for MINI switches 1-10
COM_B = "COM14" # port for MINI switches 11-20
COM_C = "COM15" # port for MINI swiches 21-28

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
time.sleep(1)

# Enable HV!!!
serial_writer.HV_enable()
time.sleep(0.5)

# prepare preprogrammed sequence:
fps = 20

output_data = haptic_map.make_output_data(generator.sawtooth(scale=0.4),
                                        freq_range=(0,50),
                                        duty_range=(0.01,0.5))

# output_data2 = haptic_map.make_output_data(generator.sine_global(total_time=20, frame_rate=10, freq=1),
#                                         freq_range=(1,10),
#                                         duty_range=(0.5,0.1))

# output_data3 = haptic_map.make_output_data(generator.ramp(total_time=5, frame_rate=10, direction=1),
#                                         freq_range=(1,100),
#                                         duty_range=(0.5,0.5))

# output_data4 = haptic_map.make_output_data(generator.ramp(total_time=5, frame_rate=10, direction=1),
#                                         freq_range=(100,100),
#                                         duty_range=(0.05,0.05))
# output_data = output_data2

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
    if t_elapsed<1/fps:
        time.sleep(1/fps-(t_elapsed)) 

    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!
    
# Disable HV!!!
serial_writer.HV_disable()
zero_output = haptic_map.make_output_data(generator.zeros())
zero_intensity, zero_packets = zero_output.pop()
serial_writer.write_packets_to_USB(zero_packets)
time.sleep(1)