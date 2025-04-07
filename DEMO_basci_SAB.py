#!/usr/bin/env python3

"""This demo script shows a basic single output sequence from the haptic display."""

###### USER SETTINGS ######
SERIAL_ACTIVE = True # if False, just runs the algorithm without sending to HV switches
COM_A = "COM10" # port for MINI switches 1-10
COM_B = "COM7" # port for MINI switches 11-20
COM_C = "COM9" # port for MINI swiches 21-28

###### INITIALIZATIONS ######
import cv2
import time
import haptic_utils.haptic_map as haptic_map
import haptic_utils.generator as generator
import haptic_utils.USB as USB
import numpy as np

###### MAIN ######

# Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)
time.sleep(1)

# Enable HV!!!
serial_writer.HV_enable()
time.sleep(0.5)

# prepare preprogrammed sequence:
fps = 10
# Define on/off grid parameters
grid_params = np.array([
    [False, True, False, False, False, True, False],
    [True, False, False, False, False, False, True],
    [False, False, False, False, False, False, False],
    # [True, True, True, True, True, True, True]
    [False, False, False, False, False, False, False]
])


output_data1 = haptic_map.make_output_data(generator.sawtooth(scale=0.4),
                                        freq_range=(0,10),
                                        duty_range=(0.01,0.5))

output_data2 = haptic_map.make_output_data(generator.sine_global(total_time=20, frame_rate=10, freq=1),
                                        freq_range=(1,10),
                                        duty_range=(0.5,0.1))

output_data3 = haptic_map.make_output_data(generator.ramp(total_time=40, frame_rate=10, direction=1),
                                        freq_range=(1,100),
                                        duty_range=(0.5,0.5))

output_data4 = haptic_map.make_output_data(generator.ramp(total_time=5, frame_rate=10, direction=1),
                                        freq_range=(100,100),
                                        duty_range=(0.05,0.05))

output_data5 = haptic_map.make_output_data(generator.sine(total_time=15,frame_rate=fps,scale=0.1, freq=2),
                                           freq_range=(10,200),
                                           duty_range=(0.5,0.5))

# output_data6 = haptic_map.make_output_data(generator.sine_global_on_off(total_time=3,frame_rate=fps,freq=2,grid_params=grid_params),
#                                              freq_range=(10,200),
#                                               duty_range=(0.5,0.5))
total_time = 50
frame_rate = 10
freq = 3
# def sine_global_on_off(total_time: float = 3, frame_rate: int = 24,freq:float=1,grid_params=None):

t = np.arange(start=0, stop=total_time, step=1/frame_rate)
output = np.zeros((4, 7, t.size))
if grid_params is None:
        grid_params = np.zeros((4, 7), dtype=bool)

# Generate the sine wave for the active pixels
for r in range(4):
    for c in range(7):
        if grid_params[r, c]:  # If "on", assign the sine wave
            output[r, c, :] = 0.5 + 0.5 * np.sin(freq*2 * np.pi * t) 
output_data6 =haptic_map.make_output_data(generator.make_output(output),duty_range=(1,1))

output_data = output_data6

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
