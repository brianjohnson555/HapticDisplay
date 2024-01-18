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

###### INITIALIZATIONS ######
## Setting up Tkinter
window = tk.Tk()
TKframe_images = []
for ii in range(0,5):
    TKframe_images.append(tk.Frame(master=window))
TKframe_labels = []
for ii in range(0,5):
    TKframe_labels.append(tk.Frame(master=window))
TKframe_entries = []
for ii in range(0,5):
    TKframe_entries.append(tk.Frame(master=window))
TKframe_buttons = [tk.Frame(master=window)]

TKframe_labels[0].grid(row=0, column=0, padx=5, pady=5)
TKframe_labels[1].grid(row=1, column=0, padx=5, pady=5)
TKframe_labels[2].grid(row=2, column=0, padx=5, pady=5)
TKframe_labels[3].grid(row=3, column=0, padx=5, pady=5)
TKframe_labels[4].grid(row=4, column=0, padx=5, pady=5, rowspan=2)
TKframe_buttons[0].grid(row=0, column=1, padx=5, pady=5)
TKframe_entries[0].grid(row=1, column=1, padx=5, pady=5)
TKframe_entries[1].grid(row=2, column=1, padx=5, pady=5)
TKframe_entries[2].grid(row=3, column=1, padx=5, pady=5)
TKframe_entries[3].grid(row=4, column=1, padx=5, pady=5)
TKframe_entries[4].grid(row=5, column=1, padx=5, pady=5)
TKframe_images[0].grid(row=0, column=2, rowspan=2)
TKframe_images[1].grid(row=0, column=3, rowspan=2)
TKframe_images[2].grid(row=2, column=2, rowspan=2)
TKframe_images[3].grid(row=2, column=3, rowspan=2)
TKframe_images[4].grid(row=4, column=2, rowspan=2)


## Setting up torch device and model, video input
device = torch.device("cpu")
dino8 = torch.hub.load('facebookresearch/dino:main','dino_vits8')
dino8.to(device)
dino8.eval()
midas_s = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
midas_transforms = torch.hub.load('intel-isl/MiDaS','transforms')
midas_s.to(device)
midas_s.eval()
video = 'boat'
cap = cv2.VideoCapture("video_" + video + ".mp4") #use video

###### HYPERPARAMETERS ######
FRAME_NUM = 1
RESOLUTION_ATT = 150
THRESHOLD_VAL = 0.1
BIAS = 0.5
DISPLAY_W = 10
DISPLAY_H = 6
ATTENTION = []
DEPTH = []
COMBINED = []
THRESHOLDED = []
DOWNSAMPLED = []

###### DEFINE FUNCTIONS ######
## Function to grab single video frame
def grab_frame(cap=cap,frame_num=1):
    global IMG
    global FRAME
    for num in range(0, frame_num):
        ret, img = cap.read()
    IMG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    FRAME = Image.fromarray(img)
    return FRAME, IMG

## Function to get self attention from video frame using DINO
def get_attention():
    global FRAME
    frame= FRAME
    global RESOLUTION_ATT
    transform1 = transforms.Compose([           
                                transforms.Resize((RESOLUTION_ATT,int(np.floor(RESOLUTION_ATT*1.7777)))),
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

def get_depth():
    global IMG
    img = IMG
    transform2 = midas_transforms.dpt_transform
    frame_t = transform2(img)
    depth = midas_s(frame_t)
    depth = depth.cpu().detach().numpy().squeeze(0)
    return depth

def get_combined():
    global BIAS
    global DEPTH
    global ATTENTION
    depth = DEPTH
    attention = ATTENTION
    depth_re = cv2.resize(depth, dsize=(720, 1280), interpolation=cv2.INTER_CUBIC)
    depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    attention_re = cv2.resize(attention, dsize=(720, 1280), interpolation=cv2.INTER_CUBIC)
    attention_nm = cv2.normalize(attention_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    combined = (depth_nm * (BIAS)*attention_nm)
    combined = cv2.normalize(combined, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return combined

def get_threshold():
    global THRESHOLD_VAL
    global COMBINED
    combined = COMBINED
    return (combined > THRESHOLD_VAL) * combined

def get_downsample():
    global DISPLAY_H
    global DISPLAY_W
    global THRESHOLDED
    threshold = THRESHOLDED
    return cv2.resize(threshold, dsize=(DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_CUBIC)

## Function to plot the self attention and original frame together
def plot_overlay(canvas, plot, data, title="default"):
    global FRAME
    canvas.draw()
    extent1 = 0, 1280, 0, 720
    plot.imshow(FRAME, extent=extent1)
    plot.imshow(data, alpha=0.6,extent=extent1)
    plot.set_title(title)
    canvas.draw()
    # data.repeat(scale,axis=0).repeat(scale,axis=1)

## Advance frame
def update_frame():
    global FRAME_NUM
    global FRAME
    FRAME_NUM += 1
    FRAME, IMG = grab_frame(frame_num=FRAME_NUM)
    update_all()

## Update all plots
def update_all():
    update_attentions()
    update_depth()
    update_combined()
    update_threshold()
    update_downsample()

## Update plot with new parameters
def update_attentions():
    global RESOLUTION_ATT
    global ATTENTION
    global CANVAS1
    ATTENTION = get_attention()
    plot_overlay(CANVAS1, plot1, ATTENTION, f"Attention (resolution={RESOLUTION_ATT})")

def update_depth():
    global DEPTH
    global CANVAS2
    DEPTH = get_depth()
    plot_overlay(CANVAS2, plot2, DEPTH, f"Depth (model=Hybrid)")

def update_combined():
    global BIAS
    global COMBINED
    global CANVAS3
    COMBINED = get_combined()
    plot_overlay(CANVAS3, plot3, COMBINED, f"Combined (bias towards attention={BIAS})")

def update_threshold():
    global THRESHOLD_VAL
    global THRESHOLDED
    global CANVAS4
    THRESHOLDED = get_threshold()
    plot_overlay(CANVAS4, plot4, THRESHOLDED, f"Threshold (cutoff:{THRESHOLD_VAL})")

def update_downsample():
    global DOWNSAMPLED
    global DISPLAY_H
    global DISPLAY_W
    global CANVAS5
    DOWNSAMPLED = get_downsample()
    plot_overlay(CANVAS5, plot5, DOWNSAMPLED, f"Downsampled (display resolution={DISPLAY_H}x{DISPLAY_W})")

###### First run of the algo ######
FRAME, IMG = grab_frame()
fig1 = Figure(figsize=(5,3))
plot1 = fig1.add_subplot(111)
CANVAS1 = FigureCanvasTkAgg(fig1, master=TKframe_images[0])
fig2 = Figure(figsize=(5,3))
plot2 = fig2.add_subplot(111)
CANVAS2 = FigureCanvasTkAgg(fig2, master=TKframe_images[1])
fig3 = Figure(figsize=(5,3))
plot3 = fig3.add_subplot(111)
CANVAS3 = FigureCanvasTkAgg(fig3, master=TKframe_images[2])
fig4 = Figure(figsize=(5,3))
plot4 = fig4.add_subplot(111)
CANVAS4 = FigureCanvasTkAgg(fig4, master=TKframe_images[3])
fig5 = Figure(figsize=(5,3))
plot5 = fig5.add_subplot(111)
CANVAS5 = FigureCanvasTkAgg(fig5, master=TKframe_images[4])
update_all()
CANVAS1.get_tk_widget().pack()
CANVAS2.get_tk_widget().pack()
CANVAS3.get_tk_widget().pack()
CANVAS4.get_tk_widget().pack()
CANVAS5.get_tk_widget().pack()

###### TKINTER LABELS AND INPUTS ######
# 
label1 = tk.Label(text="Advance frame", master=TKframe_labels[0])
label1.pack()
label2 = tk.Label(text="Set attention resolution", master=TKframe_labels[1])
label2.pack()
label3 = tk.Label(text="Set bias value (0-1)", master=TKframe_labels[2])
label3.pack()
label4 = tk.Label(text="Set threshold value (0-1)", master=TKframe_labels[3])
label4.pack()
label5 = tk.Label(text="Set display grid (W x H)", master=TKframe_labels[4])
label5.pack()


# Set up user input
entry1 = tk.Entry(width=5, master=TKframe_entries[0])
entry1.pack()
entry2 = tk.Entry(width=5, master=TKframe_entries[1])
entry2.pack()
entry3 = tk.Entry(width=5, master=TKframe_entries[2])
entry3.pack()
entry4 = tk.Entry(width=5, master=TKframe_entries[3])
entry4.pack()
entry5 = tk.Entry(width=5, master=TKframe_entries[4])
entry5.pack()

# Advance frame button
next_button = tk.Button(text="Next frame", master=TKframe_buttons[0])
next_button.pack()

###### EVENT HANDLES ######
# Do this when Enter key is pressed on the Entry item
def handle_keypress_att(entry):
    global RESOLUTION_ATT
    print("Here I am!")
    if not entry1.get():
        pass
    else:
        RESOLUTION_ATT = int(entry1.get())
    update_attentions()
    update_combined()
    update_threshold()
    update_downsample()

def handle_keypress_bias(entry):
    global BIAS
    if not entry2.get():
        pass
    else:
        BIAS = float(entry2.get())
    update_combined()
    update_threshold()
    update_downsample()

def handle_keypress_thresh(entry):
    global THRESHOLD_VAL
    if not entry3.get():
        pass
    else:
        THRESHOLD_VAL = float(entry3.get())
    update_threshold()
    update_downsample()

def handle_keypress_disp_W(entry):
    global DISPLAY_W
    if not entry4.get():
        pass
    else:
        DISPLAY_W = int(entry4.get())
    update_downsample()

def handle_keypress_disp_H(entry):
    global DISPLAY_H
    if not entry5.get():
        pass
    else:
        DISPLAY_H = int(entry5.get())
    update_downsample()

entry1.bind("<Return>", handle_keypress_att)
entry2.bind("<Return>", handle_keypress_bias)
entry3.bind("<Return>", handle_keypress_thresh)
entry4.bind("<Return>", handle_keypress_disp_W)
entry5.bind("<Return>", handle_keypress_disp_H)

# Do this when the "Next frame" button is pressed:
def handle_buttonpress(event):
    update_frame()
next_button.bind("<Button>", handle_buttonpress)


###### RUN TKINTER ######
window.mainloop()