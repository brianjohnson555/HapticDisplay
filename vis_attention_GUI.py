## Importing packages
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #interface matplotlib with tkinter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

################## INITIALIZATIONS ##################
## Setting up Tkinter
window = tk.Tk()
window.tk.call('tk', 'scaling', 0.8)

TKframe_images = [] # Create empty frames for figures
for ii in range(0,7):
    TKframe_images.append(tk.Frame(master=window))

TKframe_labels = [] # Create empty frames for labels
for ii in range(0,5):
    TKframe_labels.append(tk.Frame(master=window))

TKframe_entries = [] # Create empty frames for data entry slots
for ii in range(0,5):
    TKframe_entries.append(tk.Frame(master=window))

TKframe_buttons = [] # Create empty frames for buttons
for ii in range(0,4):
    TKframe_buttons.append(tk.Frame(master=window))

## Place all frames in the grid
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
TKframe_images[5].grid(row=4, column=3, rowspan=2)
TKframe_images[6].grid(row=0, column=4, rowspan=2)

## Define figure plots
figlist = []
plotlist = []
canvaslist = []
for fignum in range(0,7): # seven figures in total (for now)
    figlist.append(Figure(figsize=(5,3)))
    plotlist.append(figlist[fignum].add_subplot(111))
    canvaslist.append(FigureCanvasTkAgg(figlist[fignum], master=TKframe_images[fignum]))

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
video = 'game'
cap = cv2.VideoCapture("video_" + video + ".mp4") #use video

################## HYPERPARAMETERS ##################
FRAME_NUM = 1 # starting frame
RESOLUTION_ATT = 150 # resolution of get_attention for DINO model
MODEL = 'hybrid' # MiDaS model type ('small', 'hybrid', 'large')
THRESHOLD_VAL = 0.25 # threshold of attention+depth combination
BIAS = 0.6 # bias towards attention for attention+depth combination
SCALE = 4 # scaling of combined array (scale*[16, 9])
DISPLAY_W = 10 # HASEL haptic display width (pixels)
DISPLAY_H = 6 # HASEL haptic display height (pixels)
ATTENTION = [] # torch tensor holding attention data
DEPTH = [] # torch tensor holding depth data
DEPTH_BEFORE = [] # torch tensor holding depth data before gradient correction
COMBINED = [] # torch tensor holding combined (depth and attention) data
THRESHOLDED = [] # torch tensor holding thresholded data
DOWNSAMPLED = [] # torch tensor holding downsampled data

################## DEFINE FUNCTIONS ##################
# Grab video frame (next frame if frame_num=1 or nth frame if =n)
def grab_frame(cap=cap,frame_num=1):
    global IMG
    global FRAME
    for num in range(0, frame_num): # iterate through the frame numbers to get the next desired frame
        ret = cap.grab()
    ret, img = cap.retrieve() # retrieve the desired frame
    IMG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color conversion
    FRAME = Image.fromarray(img) # convert data type
    return FRAME, IMG

# Get self attention from video frame using DINO
def get_attention():
    global FRAME
    frame= FRAME
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
def get_depth():
    global IMG
    global MODEL
    img = IMG
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

    return depth, depth_sub

# combine depth and attention maps together
def get_combined(method='sum'):
    global BIAS
    global DEPTH
    global ATTENTION
    global SCALE
    depth = DEPTH
    attention = ATTENTION
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
def get_threshold():
    global THRESHOLD_VAL
    global COMBINED
    combined = COMBINED
    return (combined > THRESHOLD_VAL) * combined

# downsample the thresholded map to the grid size of the haptic display
def get_downsample():
    global DISPLAY_H
    global DISPLAY_W
    global THRESHOLDED
    threshold = THRESHOLDED
    return cv2.resize(threshold, dsize=(DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)

# Function to plot the self attention and original frame together
def plot_overlay(canvas, plot, data, title="default"):
    global FRAME
    canvas.draw()
    extent1 = 0, 1280, 0, 720
    plot.imshow(FRAME, extent=extent1)
    plot.imshow(data, alpha=0.8,extent=extent1)
    plot.set_title(title)
    canvas.draw()

# Plot just one set of data (not overlaid)
def plot_single(canvas, plot, data, title="default"):
    canvas.draw()
    extent1 = 0, 1280, 0, 720
    plot.imshow(data, extent=extent1)
    plot.set_title(title)
    canvas.draw()

# Advance frame by certain number of steps (default=1)
def update_frame(frame_step=1):
    global FRAME_NUM
    global FRAME
    FRAME_NUM += frame_step
    FRAME, IMG = grab_frame(frame_num=frame_step)
    update_all()

# Update all plots
def update_all():
    update_attentions()
    update_depth()
    update_combined()
    update_threshold()
    update_downsample()
    plot_single(canvaslist[5], plotlist[5], IMG,"Original frame")

# Update attention plot with new results
def update_attentions():
    global RESOLUTION_ATT
    global ATTENTION
    ATTENTION = get_attention()
    plot_overlay(canvaslist[0], plotlist[0], ATTENTION, f"STEP1: Attention (resolution={RESOLUTION_ATT})")

# update depth plot with new results
def update_depth():
    global DEPTH
    global MODEL
    global DEPTH_BEFORE
    DEPTH_BEFORE, DEPTH = get_depth()
    plot_overlay(canvaslist[1], plotlist[1], DEPTH_BEFORE, f"STEP2: Depth (model={MODEL})")
    plot_overlay(canvaslist[6], plotlist[6], DEPTH, f"STEP3: Depth correction")

# update combined plot with new results
def update_combined():
    global BIAS
    global COMBINED
    COMBINED = get_combined()
    plot_overlay(canvaslist[2], plotlist[2], COMBINED, f"STEP4: Combined (bias towards attention={BIAS})")

# update thresholded plot with new results
def update_threshold():
    global THRESHOLD_VAL
    global THRESHOLDED
    THRESHOLDED = get_threshold()
    plot_overlay(canvaslist[3], plotlist[3], THRESHOLDED, f"STEP5: Threshold (cutoff:{THRESHOLD_VAL})")

# update downsampled plot with new results
def update_downsample():
    global DOWNSAMPLED
    global DISPLAY_H
    global DISPLAY_W
    DOWNSAMPLED = get_downsample()
    plot_overlay(canvaslist[4], plotlist[4], DOWNSAMPLED, f"STEP6: Downsampled (display resolution={DISPLAY_H}x{DISPLAY_W})")

# save outputs from all plots
def save_output():
    global IMG
    global FRAME_NUM
    global ATTENTION
    global COMBINED
    global THRESHOLDED
    global DEPTH
    global DEPTH_BEFORE
    global DOWNSAMPLED
    datadict = {"image": IMG, "attention": ATTENTION, "depth_cor": DEPTH, "depth": DEPTH_BEFORE, "combined": COMBINED, "thresholded": THRESHOLDED, "downsampled": DOWNSAMPLED}
    
    for label in list(datadict):
        data_re = cv2.resize(datadict[label], dsize=(1280, 720), interpolation=cv2.INTER_NEAREST)
        filename = "OutputImages/output_" + str(FRAME_NUM) + "_" + label + ".png"
        # img = Image.fromarray(datadict[label].detach().numpy()[0])
        plt.imsave(filename, data_re)

################## First run of the algo ##################
FRAME, IMG = grab_frame()
update_all()
for fignum in range(0,len(figlist)):
    canvaslist[fignum].get_tk_widget().pack()

################## TKINTER LABELS AND INPUTS ###################
# Set up labels
labels = []
labels.append(tk.Label(text="Advance frame", master=TKframe_labels[0]))
labels.append(tk.Label(text="Set attention resolution", master=TKframe_labels[1]))
labels.append(tk.Label(text="Set bias value (0-1)\nSet scale factor", master=TKframe_labels[2]))
labels.append(tk.Label(text="Set threshold value (0-1)", master=TKframe_labels[3]))
labels.append(tk.Label(text="Set display grid (W x H)", master=TKframe_labels[4]))
for label in range(0, len(labels)):
    labels[label].pack()

# Set up user input with entries
entry = []
for entrynum in range(0, len(TKframe_entries)):
    entry.append(tk.Entry(width=5, master=TKframe_entries[entrynum]))
    entry[entrynum].pack()
entry.append(tk.Entry(width=5, master=TKframe_entries[1]))
entry[5].pack()

# Advance frame button
next_button = tk.Button(text="Next frame", master=TKframe_buttons[0])
next_button.pack()
next_10_button = tk.Button(text="Next 10th frame", master=TKframe_buttons[0])
next_10_button.pack()
save_button = tk.Button(text="Save outputs", master=TKframe_buttons[0])
save_button.pack()

# MiDaS model select buttons (place in Attention entry frame)
midas_l_button = tk.Button(text="MiDas Large", master=TKframe_entries[0])
midas_l_button.pack()
midas_h_button = tk.Button(text="MiDas Hybrid", master=TKframe_entries[0])
midas_h_button.pack()
midas_s_button = tk.Button(text="MiDas Small", master=TKframe_entries[0])
midas_s_button.pack()

################## EVENT HANDLES ##################
### Do this when Enter key is pressed on the Entry item:
# for attention resolution
def handle_keypress_att(entry1):
    global RESOLUTION_ATT
    if not entry[0].get(): # if user hasn't input anything, do nothing
        pass
    else:
        RESOLUTION_ATT = int(entry[0].get())
    update_attentions()
    update_combined()
    update_threshold()
    update_downsample()

# for bias
def handle_keypress_bias(entry1):
    global BIAS
    if not entry[1].get():
        pass
    else:
        BIAS = float(entry[1].get())
    update_combined()
    update_threshold()
    update_downsample()

# for scale
def handle_keypress_scale(entry1):
    global SCALE
    if not entry[5].get():
        pass
    else:
        SCALE = int(entry[5].get())
    update_combined()
    update_threshold()
    update_downsample()

# for threshold value
def handle_keypress_thresh(entry1):
    global THRESHOLD_VAL
    if not entry[2].get():
        pass
    else:
        THRESHOLD_VAL = float(entry[2].get())
    update_threshold()
    update_downsample()

# for haptic display width
def handle_keypress_disp_W(entry1):
    global DISPLAY_W
    if not entry[3].get():
        pass
    else:
        DISPLAY_W = int(entry[3].get())
    update_downsample()

# for haptic display height
def handle_keypress_disp_H(entry1):
    global DISPLAY_H
    if not entry[4].get():
        pass
    else:
        DISPLAY_H = int(entry[4].get())
    update_downsample()

# bind all entries to the events
entry[0].bind("<Return>", handle_keypress_att)
entry[1].bind("<Return>", handle_keypress_bias)
entry[2].bind("<Return>", handle_keypress_thresh)
entry[3].bind("<Return>", handle_keypress_disp_W)
entry[4].bind("<Return>", handle_keypress_disp_H)
entry[5].bind("<Return>", handle_keypress_scale)

# Do this when the "Next frame" button is pressed:
def handle_buttonpressnext(event):
    update_frame(1)
next_button.bind("<Button>", handle_buttonpressnext) # bind to Next Frame button

def handle_buttonpressnext10(event):
    update_frame(10)
next_10_button.bind("<Button>", handle_buttonpressnext10) # bind to Next 10th Frame button

# Do this when "Save output" button is pressed:
def handle_buttonpresssave(event):
    save_output()
save_button.bind("<Button>", handle_buttonpresssave) # bind to Save  button


# Do this when the MiDaS model buttons are pressed:

def handle_buttonpress_midas_l(event):
    global MODEL
    MODEL = 'large'
    update_depth()
midas_l_button.bind("<Button>", handle_buttonpress_midas_l) # bind to button

def handle_buttonpress_midas_h(event):
    global MODEL
    MODEL = 'hybrid'
    update_depth()
midas_h_button.bind("<Button>", handle_buttonpress_midas_h) # bind to button

def handle_buttonpress_midas_s(event):
    global MODEL
    MODEL = 'small'
    update_depth()
midas_s_button.bind("<Button>", handle_buttonpress_midas_s) # bind to button

################## RUN TKINTER ##################
window.mainloop()