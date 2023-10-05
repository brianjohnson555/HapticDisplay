import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import timm
import dino
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import urllib.request
import keyboard
import matplotlib.animation as animation
from time import sleep
# matplotlib.use("Agg")

device = torch.device("cpu")

#model = timm.create_model('vit_small_patch16_224_dino',pretrained=True)
model = torch.hub.load('facebookresearch/dino:main','dino_vits8')
# model = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')


# Load MiDaS model onto CPU
device = torch.device('cpu')
# midas = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
# midas = torch.hub.load('intel-isl/MiDaS', 'custom', path='../utilites/dpt_beit_large_512.pt', force_reload=True)
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def get_last_self_attention(self, x, masks=None):
    if isinstance(x, list):
        return self.forward_features_list(x, masks)
        
    x = self.prepare_tokens_with_masks(x, masks)
    
    # Run through model, at the last block just return the attention.
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else: 
            return blk(x, return_attention=True)


# Setup frame capture
cap = cv2.VideoCapture('video4.mp4') #use video
#cap = cv2.VideoCapture(0) #stream from webcam
previous_frame = None

frames_list = []
attentions_list = []
depth_list = []
combined_list = []
Hasel_list = []

while True: 
    # Load frame
    ret= cap.grab()
    if ret is False:
        break
    ret, img = cap.retrieve()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = img
    # img = cv2.addWeighted(img,1.5,img,0,1)
    
    image = Image.fromarray(img)
    Tx = transforms.Resize((20*9,20*16))(image)
    Tx2 = transforms.ToTensor()(Tx).unsqueeze_(0)
    Tx3 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(Tx2)
    Tx3.requires_grad = True

    model.eval()
    model.to(device)

    start_time = time.time()
    attentions = model.get_last_selfattention(Tx3)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    nh = attentions.shape[1]
    attentions = attentions[0, :, 0, 1:].reshape(nh,-1)
    patch_size = 4
    w_featmap = Tx3.shape[-2] // patch_size
    h_featmap = Tx3.shape[-1] // patch_size

    attentions = attentions.reshape(nh, w_featmap//2, h_featmap//2)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=1, mode="nearest")[0].detach().numpy()
    attentions_mean = np.mean(attentions, axis=0)


    # Compute depth:
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(20*9, 20*16),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    
    new_size = attentions_mean.shape

    depth_re = cv2.resize(depth, dsize=(new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)
    depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    attentions_nm = cv2.normalize(attentions_mean, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)

    combined = depth_nm * attentions_nm
    combined = (combined > 0.1) * combined
    Hasel = cv2.resize(combined, dsize=(10, 6), interpolation=cv2.INTER_CUBIC)

    # Update:
    attentions_list.append(attentions_mean)
    depth_list.append(depth)
    frames_list.append(frame)
    combined_list.append(combined)
    Hasel_list.append(Hasel)

ims = []
figure = plt.figure()
for i in range(0,len(frames_list)):
    im = plt.imshow(attentions_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
ani.save('animation4_attention20.mp4', writer = 'ffmpeg', bitrate=1000, fps=15)
plt.close()

ims = []
figure = plt.figure()
for i in range(0,len(frames_list)):
    im = plt.imshow(frames_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
ani.save('animation4_frames.mp4', writer = 'ffmpeg', bitrate=1000, fps=15)
plt.close()

ims = []
figure = plt.figure()
for i in range(0,len(frames_list)):
    im = plt.imshow(depth_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
ani.save('animation4_depth20.mp4', writer = 'ffmpeg', bitrate=1000, fps=15)
plt.close()

ims = []
figure = plt.figure()
for i in range(0,len(frames_list)):
    im = plt.imshow(combined_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
ani.save('animation4_combined2020.mp4', writer = 'ffmpeg', bitrate=1000, fps=15)
plt.close()

ims = []
figure = plt.figure()
for i in range(0,len(frames_list)):
    im = plt.imshow(Hasel_list[i], animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
ani.save('animation4_HASEL_10_6.mp4', writer = 'ffmpeg', bitrate=1000, fps=15)
plt.close()

# input('Ready to display')
# plt.show()
