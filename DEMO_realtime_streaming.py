#!/usr/bin/env python3

"""This demo script runs an example of real-time visual-haptic streaming.

The script creates a video of the computer webcam view, which should be lined up to 
match the position of the haptic display on the screen. The algorithm estimates a 
depth-map of each frame of the webcam image, and the mapping function maps depth to
frequency/duty cycle."""

###### USER SETTINGS ######
SERIAL_ACTIVE = True # if False, just runs the algorithm without sending to HV switches
VIEW_INTENSITY = False # if True, opens a video showing the depth/intensity map
SAVE_VIDEO = False # if True, saves a video file of both the camera view and the intensity map
COM_A = "COM9" # port for MINI switches 1-10
COM_B = "COM14" # port for MINI switches 11-20
COM_C = "COM15" # port for MINI swiches 21-28

###### INITIALIZATIONS ######
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import haptic_utils.haptic_map as haptic_map
import haptic_utils.USB as USB # my custom file

###### MAIN ######
# TODO: Need to make all of this work with new algo_functions codes.

def map_pixels(output):
    periods_bot = np.array([output[1,0], output[0,0], output[1,1], output[0,1], output[1,2], output[0,2], output[1,3], output[0,3], output[1,4], output[0,4]])
    periods_top = np.array([output[3,0], output[2,0], output[3,1], output[2,1], output[3,2], output[2,2], output[3,3], output[2,3], output[3,4], output[2,4]])
    return periods_bot, periods_top

# Load MiDaS model onto CPU
device = torch.device('cpu')
midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Set up frame capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #stream from webcam

# Set up USB:
serial_ports = [COM_A, COM_B, COM_C]
serial_writer = USB.SerialWriter(serial_ports, serial_active=SERIAL_ACTIVE)

if SAVE_VIDEO:
    outlist = []
    outlist2 = []

# Enable HV!!!
serial_writer.HV_enable()
    
while True: 
    # Load frame
    ret,img = cap.read()
    imgcolor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgcolor = cv2.addWeighted(imgcolor,1.5,imgcolor,0,1)

    # Compute depth
    frame = midas_transforms.small_transform(imgcolor) # use small image transform
    depth_size_x = 256
    depth = midas(frame) # evaluate using small model
    depth = depth.cpu().detach().numpy().squeeze(0)

    # remove normal depth gradient from depth map
    depth_nm = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    xgrid = np.zeros(depth_size_x, dtype=float)
    ygrid = depth_nm.mean(axis=1) # make mean-depth-based gradient
    grad_array = np.meshgrid(xgrid, ygrid)[1] # form gradient array w/ same size as depth
    depth_sub = (depth_nm - grad_array)
    depth_sub = (depth_sub > 0) * depth_sub # take only positive values

    # resize and threshold
    depth_re = cv2.resize(depth_sub, dsize=(1*7, 1*4), interpolation=cv2.INTER_CUBIC)
    depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    threshold = 0.26
    algo_output = (depth_nm > threshold) * depth_nm
    
    # Show frames
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Video',img)
    if VIEW_INTENSITY:
        cv2.namedWindow('Output',cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Output',algo_output)
    if SAVE_VIDEO:
        outlist.append(imgcolor)
        outlist2.append(algo_output)

    duty_array, period_array = haptic_map.linear_map_single(algo_output) # map from algo intensity to duty cycle/period
    serial_writer.write_array_to_USB(duty_array, period_array)

    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break # BREAK OUT OF LOOP WHEN "B" KEY IS PRESSED!

if SAVE_VIDEO:
    import matplotlib.animation as animation
    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist)):
        im = plt.imshow(outlist[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "OutputVideos/streaming_output_cam.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=10)
    plt.close()

    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist2)):
        im = plt.imshow(outlist2[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "OutputVideos/streaming_output_algo.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=10)
    plt.close()

# Disable HV!!!
serial_writer.HV_disable()
time.sleep(0.5)