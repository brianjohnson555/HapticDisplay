#!/usr/bin/env python3

"""This example script shows you how to use the functions in this repository to build haptic outputs."""

###### USER SETTINGS ######
FILENAME = "algo_input_data/datakoi_output.txt" # file to load for output data
VIDEONAME = "algo_input_videos/video_koi.mp4"
SERIAL_ACTIVE = False # if False, just runs the algorithm without sending to HV switches
FRAME_RATE = 60 # frame rate (fps) to run video
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

### Set up serial port objects
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)
time.sleep(0.5) # wait so that serial ports can initialize
serial_writer.HV_enable() # Enable HV!!!

### Prepare data for loop:
# You have two options: (1) generate your own haptic output using generator.py functions
#                       (2) load haptic intensities from a file (created using algo_preprocessing.py)

### Option (1): generating outputs:
# add preprogrammed sequence with all options specified:
output_data_1 = haptic_map.make_output_data(generator.ramp(total_time=5,
                                                           frame_rate=FRAME_RATE,
                                                           direction=-1),
                                        freq_range=(0,200),
                                        duty_range=(0.05,0.5))
# add another preprogrammed sequence, this time using default arguments:
output_data_2 = haptic_map.make_output_data(generator.sine_global())
# append the extra sequence onto the original sequence:
output_data_1.extend(output_data_2)

### Option (2): load outputs from file:
from numpy import genfromtxt
data=genfromtxt(FILENAME,delimiter=',')[:,0:28] # load from file
output_data = haptic_map.make_output_data(data,
                                        freq_range=(0,24),
                                        duty_range=(0.1,0.5))
# Preprocess video (should correspond with outputs from file):
video_data = []
cap = cv2.VideoCapture(VIDEONAME)
while True:
    ret,img = cap.read()
    if ret:
        video_data.append(img)
    else:
        break
video_data.reverse()

### Run real-time loop to display video and send haptic commands to USB:
while True:
    isClose=False # break condition
    video_sequence = video_data.copy() # copy video sequence
    packet_sequence = output_data.packet_sequence.copy() # copy packet sequence
    while True:
        if len(packet_sequence)>0 and len(video_sequence)>0: # if frame exists, run; otherwise, video is finished->loop back to beginning
            t_start = time.time()
            # send to USB:
            packets = packet_sequence.pop()
            serial_writer.write_packets_to_USB(packets)
            # Display video:
            img = video_sequence.pop()
            cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video',img)
            # get elapsed time:
            t_end=time.time()
            t_elapsed = t_end-t_start
            # maintain constant loop frame rate:
            if t_elapsed<1/FRAME_RATE:
                time.sleep(1/FRAME_RATE-(t_elapsed)) 

            if(cv2.waitKey(10) & 0xFF == ord('b')): # BREAK OUT OF LOOP WHEN "b" KEY IS PRESSED!
                isClose = True # assign stop flag
                break
        else: 
            break # video is finished, break and reset frame count

    if isClose: 
        break # user pressed 'b', stop script
    

serial_writer.HV_disable() # Disable HV when finished!!!
time.sleep(0.5) # pause to let HV disable before script ends and serial connection closes
