#!/usr/bin/env python3

"""This script processes input video through the visual-haptic algorithm.

NOTE: Work in progress!"""

###### USER SETTINGS ######
VIDEO = "koi" # pick video suffix from algo_input_videos/ folder
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
cap = cv2.VideoCapture("algo_input_videos/video_" + VIDEO + ".mp4") #use video
device = torch.device(DEVICE)
model = algo.VisualHapticModel(device=device,
                               resolution_attention=RESOLUTION_ATT,
                               depth_model_type=MODEL,
                               threshold_value=THRESHOLD_VAL,
                               bias=BIAS,
                               scaling_factor=SCALE,
                               display_dim=DISPLAY_DIMS)

frame_num = 1
output_list = []
depth_list = []
attention_list = []


print("Running...")
while True: 
        # grab next frame from video
        frame = algo.grab_video_frame(cap)
        if frame is None:
            break # no more frames, finish loop
        
        attention = model.get_attention(frame)
        depth_frame = model.get_depth(frame)
        depth_grad = model.remove_gradient(depth_frame)

        combined_frame = model.get_combined(depth_grad, attention)
        threshold_frame = model.get_threshold(combined_frame)
        output = model.get_downsample(threshold_frame)

        output_list.append(output)
        depth_list.append(depth_grad)
        attention_list.append(attention)
        print(len(output_list))
        frame_num += 1

print("Global scaling...")
attention_np = np.array(attention_list)
flat = attention_np.flatten()
scale_att = 1/(np.mean(flat)+3*np.std(flat))

depth_np = np.array(depth_list)
flat = depth_np.flatten()
scale_dep = 1/(np.mean(flat)+3*np.std(flat))

output_list_scaled = []
BIAS = 0.9

for frame in range(len(attention_list)):
    attention_nm = attention_list[frame]*scale_att
    depth_nm = depth_list[frame]*scale_dep

    depth_re = cv2.resize(depth_nm, dsize=(16*SCALE, 9*SCALE), interpolation=cv2.INTER_CUBIC)
    attention_re = cv2.resize(attention_nm, dsize=(16*SCALE, 9*SCALE), interpolation=cv2.INTER_CUBIC)

    # combined = ((1-BIAS)*depth_re + (BIAS)*attention_re)
    combined = depth_re*attention_re
    thresholded = (combined > THRESHOLD_VAL) * combined

    ### new code for interpolation:
    downsampled = np.zeros((DISPLAY_DIMS[0], DISPLAY_DIMS[1]))
    interval_H = int(np.floor(thresholded.shape[0]/DISPLAY_DIMS[0]))
    interval_W = int(np.floor(thresholded.shape[1]/DISPLAY_DIMS[1]))
    for rr in range(0, DISPLAY_DIMS[0]):
        for cc in range(0, DISPLAY_DIMS[1]):
            frame_slice = thresholded[rr*interval_H:(rr+1)*interval_H, cc*interval_W:(cc+1)*interval_W]
            mean_slice = np.mean(frame_slice)
            std_slice = np.std(frame_slice)
            max_slice = np.max(frame_slice)
            if mean_slice+3*std_slice > max_slice: # I'm doing some weird selection of max vs. mean depending on std of the frame slice
                downsampled[rr, cc] = max_slice
            else:
                downsampled[rr, cc] = mean_slice

    output_list_scaled.append(downsampled)

output_np = np.array(output_list_scaled)
flat = output_np.flatten()
max_thresh = (np.mean(flat)+5*np.std(flat))
output_np[output_np>max_thresh] = max_thresh
scale_output = 1/(max_thresh)
output_scale = output_np*scale_output

for frame in range(len(output_list_scaled)):
    output_list_scaled[frame] = output_scale[frame]

ims = []
figure = plt.figure()
for i in range(0,len(output_list)):
    im = plt.imshow(output_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "output_videos/animation_" + VIDEO + "_output_original.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=30)
plt.close()

figure = plt.figure()
for i in range(0,len(output_list_scaled)):
    im = plt.imshow(output_list_scaled[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "output_videos/animation_" + VIDEO + "_output_scaled.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=30)
plt.close()

filename_data = "algo_input_data/data" + VIDEO + "_output_scaled.txt"
with open(filename_data, 'w') as fo:
    for idx, item in enumerate(output_list_scaled):
        for row in range(DISPLAY_DIMS[0]):
            for column in range(DISPLAY_DIMS[1]):
                fo.write(str(item[row, column]) + ', ')
        fo.write('\n')

print("Finished!")