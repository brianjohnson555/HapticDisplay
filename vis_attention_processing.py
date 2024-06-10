import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.animation as animation

## Setting up torch device and model, video input
device = torch.device("cpu")
dino8 = torch.hub.load('facebookresearch/dino:main','dino_vits8')
dino8.to(device)
dino8.eval()
midas_h = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
midas_l = torch.hub.load('intel-isl/MiDaS','DPT_Large')
midas_s = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas_transforms = torch.hub.load('intel-isl/MiDaS','transforms')
midas_h.to(device)
midas_h.eval()
midas_l.to(device)
midas_l.eval()
midas_s.to(device)
midas_s.eval()
video = 'truck'
cap = cv2.VideoCapture("video_" + video + ".mp4") #use video

frame_num = 1
output_list = []

################## HYPERPARAMETERS ##################
RESOLUTION_ATT = 150 # resolution of get_attention for DINO model
MODEL = 'hybrid' # MiDaS model type ('small', 'hybrid', 'large')
THRESHOLD_VAL = 0.25 # threshold of attention+depth combination
BIAS = 0.6 # bias towards attention for attention+depth combination
SCALE = 4 # scaling of combined array (scale*[16, 9])
DISPLAY_W = 5 # HASEL haptic display width (pixels)
DISPLAY_H = 4 # HASEL haptic display height (pixels)

################## FUNCTIONS ##################
# Grab video frame (next frame if frame_num=1 or nth frame if =n)
def grab_frame(cap=cap):
    frame = None
    img = None
    ret = cap.grab()
    if ret is False:
        return frame, img
    ret, img = cap.retrieve() # retrieve the desired frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color conversion
    frame = Image.fromarray(img) # convert data type
    return frame, img

# Get self attention from video frame using DINO
def get_attention(frame):
    global RESOLUTION_ATT
    transform1 = transforms.Compose([           
                                transforms.Resize((RESOLUTION_ATT,int(np.floor(RESOLUTION_ATT*1.7777)))),
                                # transforms.CenterCrop(resolution), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                #transforms.Normalize(mean=0.5, std=0.2)
                                ])
    frame_t = transform1(frame).unsqueeze(0)
    attentions = dino8.get_last_selfattention(frame_t)
    nh = attentions.shape[1]
    attentions = attentions[0, :, 0, 1:].reshape(nh,-1)
    patch_size = 4
    w_featmap = frame_t.shape[-2] // patch_size
    h_featmap = frame_t.shape[-1] // patch_size

    attentions = attentions.reshape(nh, w_featmap//2, h_featmap//2)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=1, mode="nearest")[0].detach().numpy()
    attentions_mean = (np.mean(attentions, axis=0))
    return attentions_mean

# get depth estimation of frame using MiDaS
def get_depth(img):
    global MODEL
    if MODEL == 'small':
        frame = midas_transforms.small_transform(img) # use small image transform
        depth_size_x = 256
        depth = midas_s(frame) # evaluate using small model
    else:
        frame = midas_transforms.dpt_transform(img) # use DPT image transform
        depth_size_x = 672
        if MODEL == 'hybrid':
            depth = midas_h(frame) # evaluate using hybrid model
        else:
            depth = midas_l(frame) # evaluate using large model

    depth = depth.cpu().detach().numpy().squeeze(0)
    
    # remove normal depth gradient from depth map
    depth_nm = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    xgrid = np.zeros(depth_size_x, dtype=float)
    ygrid = depth_nm.mean(axis=1) # make mean-depth-based gradient
    grad_array = np.meshgrid(xgrid, ygrid)[1] # form gradient array w/ same size as depth
    depth_sub = (depth_nm - grad_array)
    depth_sub = (depth_sub > 0) * depth_sub # take only positive values

    return depth_sub

# combine depth and attention maps together
def get_combined(depth, attention, method='sum'):
    global BIAS
    global SCALE
    depth_re = cv2.resize(depth, dsize=(16*SCALE, 9*SCALE), interpolation=cv2.INTER_CUBIC)
    depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    attention_re = cv2.resize(attention, dsize=(16*SCALE, 9*SCALE), interpolation=cv2.INTER_CUBIC)
    attention_nm = cv2.normalize(attention_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)

    if method=='multiply':
        combined = depth_nm*attention_nm
    elif method=='sum':
        combined = ((1-BIAS)*depth_nm + (BIAS)*attention_nm)
    combined = cv2.normalize(combined, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return combined

# threshold the combined map to threshold_val
def get_threshold(combined):
    global THRESHOLD_VAL
    return (combined > THRESHOLD_VAL) * combined

# downsample the thresholded map to the grid size of the haptic display
def get_downsample(thresholded):
    global DISPLAY_H
    global DISPLAY_W
    return cv2.resize(thresholded, dsize=(DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)


print("Running...")
while True: 
        frame, img = grab_frame(cap)
        if frame is None:
            break
        attention = get_attention(frame)
        depth = get_depth(img)
        combined = get_combined(depth, attention)
        thresholded = get_threshold(combined)
        output = get_downsample(thresholded)

        frame_num += 1
        print(frame_num)
        output_list.append(output)

print("Plotting...")

ims = []
figure = plt.figure()
for i in range(0,len(output_list)):
    im = plt.imshow(output_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
filename = "animation_" + video + "_output.mp4"
ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=15)
plt.close()

filename_data = "data" + video + "_output.txt"
with open(filename_data, 'w') as fo:
    for idx, item in enumerate(output_list):
        for row in range(DISPLAY_H):
            for column in range(DISPLAY_W):
                fo.write(str(item[row, column]) + ', ')
        fo.write('\n')

print("Finished!")