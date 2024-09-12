#!/usr/bin/env python3

"""This script processes input video through the visual-haptic algorithm.

NOTE: Work in progress!"""

###### USER SETTINGS ######
VIDEO = "truck" # pick video suffix from algo_input_videos/ folder
RESOLUTION_ATT = 100 # resolution of get_attention for DINO model
MODEL = 'hybrid' # MiDaS model type ('small', 'hybrid', 'large')
THRESHOLD_VAL = 0.35 # threshold of attention+depth combination
BIAS = 0.75 # bias towards attention for attention+depth combination
SCALE = 4 # scaling of combined array (scale*[16, 9])
DISPLAY_DIMS = (4,7) # HASEL haptic display dimensions, H x W (pixels)
FRAME_SKIP = 5 # interval for how often to calculate algorithm (then interpolate between)
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
                               depth_model_type=MODEL,
                               threshold_value=THRESHOLD_VAL,
                               bias=BIAS,
                               scaling_factor=SCALE,
                               display_dim=DISPLAY_DIMS)

frame_num = 1
output_list = []

print("Running...")
while True: 
        # grab next frame from video
        frame = algo.grab_video_frame(cap)

        if frame is None:
            break # no more frames, finish loop
        if frame_num%FRAME_SKIP==1: #### ONLY PROCESS EVERY x FRAMES!

            output = model.single_run(frame) # run the visual-haptic algorithm

            if output_list:#skip first iteration
                last_output = output_list[-1]
                for frame in range(1, FRAME_SKIP):
                    interp_vec = np.zeros(DISPLAY_DIMS)
                    for row in range(DISPLAY_DIMS[0]):
                        for col in range(DISPLAY_DIMS[1]):
                            interp_vec[row, col] = np.linspace(last_output[row, col], output[row, col], FRAME_SKIP+1)[frame]
                    output_list.append(interp_vec)

            output_list.append(output)
            print(len(output_list))

        frame_num += 1

print("Plotting...")

ims = []
figure = plt.figure()
for i in range(0,len(output_list)):
    im = plt.imshow(output_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "output_videos/animation_" + VIDEO + "_output.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=30)
plt.close()

filename_data = "algo_input_data/data" + VIDEO + "_output.txt"
with open(filename_data, 'w') as fo:
    for idx, item in enumerate(output_list):
        for row in range(DISPLAY_DIMS[0]):
            for column in range(DISPLAY_DIMS[1]):
                fo.write(str(item[row, column]) + ', ')
        fo.write('\n')

print("Finished!")