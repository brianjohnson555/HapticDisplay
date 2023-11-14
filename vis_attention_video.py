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
dino8 = torch.hub.load('facebookresearch/dino:main','dino_vits8')
dino8.to(device)
dino8.eval()
dino16 = torch.hub.load('facebookresearch/dino:main','dino_vits16')
dino16.to(device)
dino16.eval()
# model = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
# midas = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
midas_s = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas_s.to(device)
midas_s.eval()
midas_h = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
midas_h.to(device)
midas_h.eval()
# midas = torch.hub.load('intel-isl/MiDaS', 'custom', path='../utilites/dpt_beit_large_512.pt', force_reload=True)
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
video = 'boat'
cap = cv2.VideoCapture("video_" + video + ".mp4") #use video
#cap = cv2.VideoCapture(0) #stream from webcam
previous_frame = None

frames_list = []
attentions_list = []
depth_list = []
combined_list = []
Hasel_list = []
imsize = [15, 30, 45]
attentions_mean = [[], [], []]
depth = [[], [], []]

while True: 
    # Load frame
    ret= cap.grab()
    if ret is False:
        break
    ret, img = cap.retrieve()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = img
    # img = cv2.addWeighted(img,1.5,img,0,1)
    
    ### Compute attention ###    
    image = Image.fromarray(img)
    input_batch = transform(img).to(device)
    for ii in range(0,len(imsize)):
        Tx = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transforms.ToTensor()(transforms.Resize((imsize[ii]*9,imsize[ii]*16))(image)).unsqueeze_(0))
        Tx.requires_grad = True

        attentions = dino8.get_last_selfattention(Tx)
    
        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 1:].reshape(nh,-1)
        patch_size = 4
        w_featmap = Tx.shape[-2] // patch_size
        h_featmap = Tx.shape[-1] // patch_size

        attentions = attentions.reshape(nh, w_featmap//2, h_featmap//2)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=1, mode="nearest")[0].detach().numpy()
        attentions_mean[ii] = np.mean(attentions, axis=0)

    ### Compute depth ###
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))

        with torch.no_grad():
            prediction = midas_s(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(imsize[ii]*9, imsize[ii]*16),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth[ii] = prediction.cpu().numpy()
    
        new_size = attentions_mean[0].shape

        depth_re = cv2.resize(depth[ii], dsize=(new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)
        depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
        attentions_nm = cv2.normalize(attentions_mean[ii], None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)

    print(type(depth_re))
    print(type(depth_nm))
    combined = depth_nm * attentions_nm
    combined = (combined > 0.1) * combined
    Hasel = cv2.resize(combined, dsize=(10, 6), interpolation=cv2.INTER_CUBIC)

    ### Update ###
    attentions_list.append(attentions_mean)
    depth_list.append(depth)
    frames_list.append(frame)
    combined_list.append(combined)
    Hasel_list.append(Hasel)
    print(len(frames_list))

data_list = [frames_list, attentions_list, depth_list, combined_list, Hasel_list]
name_list = ["frames", "attention", "depth", "combined", "HASEL"]

for i in range(0,len(data_list)):
    print("Plotting %i list data" % (i))
    ims = []
    figure = plt.figure()
    for j in range(0,len(frames_list)):
        im = plt.imshow(data_list[i][j], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "animation_" + video + "_" + name_list[i] + ".mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=15)
    plt.close()

print("Finished!")