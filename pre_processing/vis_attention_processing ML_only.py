#!/usr/bin/env python3

"""This script processes input video through the visual-haptic algorithm. This only
runs the video frames through the DINO and MiDaS models, it does not do any other processing.

NOTE: Work in progress!"""

###### USER SETTINGS ######
VIDEO = "truck" # pick video suffix from algo_input_videos/ folder
RESOLUTION_ATT = 100 # resolution of get_attention for DINO model
MODEL = 'hybrid' # MiDaS model type ('small', 'hybrid', 'large')
FRAME_RATE = 30 # video frame rate (must check video properties!!)
DEVICE = "cpu"

###### INITIALIZATIONS ######
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import matplotlib.animation as animation
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import haptic_utils.algo_preprocessing as algo

###### MAIN ######
## Setting up torch device and model, video input
device = torch.device(DEVICE)
cap = cv2.VideoCapture("algo_input_videos/video_" + VIDEO + ".mp4") #use video
model = algo.VisualHapticModel(device=device,
                               resolution_attention=RESOLUTION_ATT,
                               depth_model_type=MODEL)

frame_num = 1
output_list = []

print("Running...")
while True: 
        # grab next frame from video
        frame = algo.grab_video_frame(cap)

        if frame is None:
            break # no more frames, finish loop
        
        model.single_run(frame) # run the visual-haptic algorithm
        output = model.output
        # append latest output to list
        output_list.append(output)
        print("Current frame= ", len(output_list))

        frame_num += 1

print("Plotting and saving...")

ims = []
figure = plt.figure()
for i in range(0,len(output_list)):
    im = plt.imshow(output_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "output_videos/animation_" + VIDEO + "_output.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=FRAME_RATE)
plt.close()

filename_data = "algo_input_data/data" + VIDEO + "_output.txt"
with open(filename_data, 'w') as fo:
    for idx, item in enumerate(output_list):
        for row in range(DISPLAY_DIMS[0]):
            for column in range(DISPLAY_DIMS[1]):
                fo.write(str(item[row, column]) + ', ')
        fo.write('\n')

print("Finished!")