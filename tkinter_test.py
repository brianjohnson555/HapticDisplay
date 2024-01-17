## Importing packages
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
window = tk.Tk()

## Setting up torch device and model, video input
device = torch.device("cpu")
dino8 = torch.hub.load('facebookresearch/dino:main','dino_vits8')
dino8.to(device)
dino8.eval()
video = 'boat'
cap = cv2.VideoCapture("video_" + video + ".mp4") #use video
RESOLUTION = 150
FRAME_NUM = 1
THRESHOLD = 0.1
BIAS = 0.5

## Function to grab single video frame
def grab_frame(cap=cap,frame_num=1):
    print(frame_num)
    for num in range(0, frame_num):
        ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    return frame

## Function to get self attention from video frame using DINO
def get_attention(frame):
    global RESOLUTION
    transform1 = transforms.Compose([           
                                transforms.Resize((RESOLUTION,int(np.floor(RESOLUTION*1.7777)))),
                                # transforms.CenterCrop(resolution), #should be multiple of model patch_size                 
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
    global RESOLUTION
    canvas.draw()
    extent1 = 0, 1280, 0, 720
    scale = 720/RESOLUTION
    extent2 = 0, 1280, 0, 720
    plot1.imshow(frame, extent=extent1)
    plot1.imshow(attentions.repeat(scale,axis=0).repeat(scale,axis=1), alpha=0.5,extent=extent1)
    canvas.draw()

## Advance frame:
def update_frame():
    global FRAME_NUM
    global FRAME
    FRAME_NUM += 1
    FRAME = grab_frame(frame_num=FRAME_NUM)
    update_plot(frame=FRAME)

## Update plot with new parameters
def update_plot(frame):
    attentions = get_attention(frame)
    plot_attention(frame, attentions)

## First run of the algo:
FRAME = grab_frame()
attentions = get_attention(FRAME)
fig = Figure()
plot1 = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=window)
plot_attention(FRAME, attentions)
canvas.get_tk_widget().pack()

## TKinter setup
TKframe = tk.Frame()
label = tk.Label(text=f"Current RESOLUTION value: {RESOLUTION}", master=TKframe)
label.pack()

# Set up user input
entry = tk.Entry(width=5, master=TKframe)
entry.pack()
name = entry.get()

# Advance frame button
next_button = tk.Button(text="Next frame", master=TKframe)
next_button.pack()

# Do this when Enter key is pressed on the Entry item
def handle_keypress(event):
    global RESOLUTION
    global FRAME
    RESOLUTION = int(entry.get())
    label["text"] = f"Current RESOLUTION value: {RESOLUTION}"
    update_plot(frame=FRAME)
entry.bind("<Return>", handle_keypress)

# Do this when the "Next frame" button is pressed:
def handle_buttonpress(event):
    update_frame()
next_button.bind("<Button>", handle_buttonpress)

## Run TKinter
TKframe.pack(fill=tk.X)
window.mainloop()