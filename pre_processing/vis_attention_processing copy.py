#!/usr/bin/env python3

"""This script processes input video through the visual-haptic algorithm.

NOTE: Work in progress!"""

###### USER SETTINGS ######
VIDEO = "street" # pick video suffix from algo_input_videos/ folder
RESOLUTION_ATT = 100 # resolution of get_attention for DINO model
MODEL = 'hybrid' # MiDaS model type ('small', 'hybrid', 'large')
THRESHOLD_VAL = 0.35 # threshold of attention+depth combination
BIAS = 0.75 # bias towards attention for attention+depth combination
SCALE = 4 # scaling of combined array (scale*[16, 9])
DISPLAY_DIMS = (4,7) # HASEL haptic display dimensions, H x W (pixels)
FRAME_SKIP = 5 # interval for how often to calculate algorithm (then interpolate between)
FRAME_RATE = 30 # video frame rate (must check video properties!!)
# note on FRAME_SKIP: only needed when you want fast processing time. Otherwise set to 0.
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
# device = torch.device(DEVICE)
# cap = cv2.VideoCapture("algo_input_videos/video_" + VIDEO + ".mp4") #use video
# model = algo.VisualHapticModel(device=device,
#                                resolution_attention=RESOLUTION_ATT,
#                                depth_model_type=MODEL,
#                                threshold_value=THRESHOLD_VAL,
#                                bias=BIAS,
#                                scaling_factor=SCALE,
#                                display_dim=DISPLAY_DIMS)

# print("Running...")

# model.full_run(cap,"street",0)

# print("Finished!")


# ## Setting up torch device and model, video input
# device = torch.device(DEVICE)
# cap = cv2.VideoCapture("algo_input_videos/video_koi.mp4") #use video
# model2 = algo.VisualHapticModel(device=device,
#                                resolution_attention=RESOLUTION_ATT,
#                                depth_model_type=MODEL,
#                                threshold_value=THRESHOLD_VAL,
#                                bias=BIAS,
#                                scaling_factor=SCALE,
#                                display_dim=DISPLAY_DIMS)

# print("Running...")

# model2.full_run(cap,"koi2",0)

# print("Finished!")


cap = cv2.VideoCapture("algo_input_videos/video_" + VIDEO + ".mp4") #use video
frame_list=[]
while True:
    frame = algo.grab_video_frame(cap)
    if frame is None:
        break # no more frames, finish loop
    frame_list.append(frame)

ims = []
figure = plt.figure()
for i in range(0,len(frame_list)):
    im = plt.imshow(frame_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "output_videos/animation_" + VIDEO + "_video.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=20)
plt.close()

cap2 = cv2.VideoCapture("algo_input_videos/video_koi.mp4") #use video
frame_list=[]
while True:
    frame = algo.grab_video_frame(cap2)
    if frame is None:
        break # no more frames, finish loop
    frame_list.append(frame)

ims = []
figure = plt.figure()
for i in range(0,len(frame_list)):
    im = plt.imshow(frame_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "output_videos/animation_" + "koi" + "_video.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=20)
plt.close()

