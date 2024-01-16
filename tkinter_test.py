## Importing packages
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

## Setting up torch device and model, video input
device = torch.device("cpu")
dino8 = torch.hub.load('facebookresearch/dino:main','dino_vits8')
dino8.to(device)
dino8.eval()
video = 'boat'
cap = cv2.VideoCapture("video_" + video + ".mp4") #use video

## Function to grab single video frame
def grab_frame(cap=cap):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    return frame

## Function to get self attention from video frame using DINO
def get_attention(frame, resolution):
    transform1 = transforms.Compose([           
                                transforms.Resize(resolution),
                                transforms.CenterCrop(resolution-2), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
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

## Function to plot the self attention and original frame together
def plot_attention(frame, attentions):
    fig, axs = plt.figure()
    axs[1,2].imshow(frame)
    axs[1,2].imshow(attentions.repeat(2,axis=0).repeat(2,axis=1), alpha=0.5)

## Function to update all parameters
def update(resolution):
    frame = grab_frame()
    attentions = get_attention(frame, resolution)
    plot_attention(frame, attentions)

## TKinter setup
window = tk.Tk()
greeting = tk.Label(text="Hello, world", background="white")
greeting.pack()
window.mainloop()