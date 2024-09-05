#!/usr/bin/env python3

"""Preprocessing functions for the visual-haptic algorithm.

This is the heart of the visual-haptic algorithm. The script defines the VisualHapticModel
class, which contains all relevant functions for processing visual image data into intensities.

The resultant intensity arrays can be passed to utils.haptic_funcs using the HapticMap class
to convert the intensities into haptic outputs of frequency and duty cycle.

Visual-haptic processing scripts are located in /pre_processing/ directory"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

# Grab video frame (next frame if frame_num=1 or nth frame if =n)
def grab_video_frame(source:cv2.VideoCapture):
    """Grabs the next frame from the video source.
    
    Inputs:
    -source: cv2.VideoCapture object which is the video source"""

    ret = source.grab()
    if ret is False: # if no data is available from the source, return Nones
        return None
    ret, frame = source.retrieve() # retrieve the desired frame from source
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # color conversion
    return frame

class VisualHapticModel:
    """Class wrapper for the visual haptic algorithm.
    
    VisualHapticModel contains all processing steps of the algorithm as separate methods.
    On __init__(), the user defines all parameters for the algorithm."""

    def __init__(self, device:torch.device, resolution_attention:int = 75, depth_model_type:str = 'hybrid', threshold_value:float = 0.25, bias:float = 0.75, combine_method:str = 'sum', scaling_factor:int = 4, display_dim:tuple = (4,7)):
        """Initialization of VisualHapticModel. Sets parameters and loads pretrained models.
        
        Inputs:
        -device: device for PyTorch ("cpu" or "cuda")
        -resolution_attention: resolution to scale the input image for attention model
        -depth_model_type: type of model for MiDaS depth ("small", "hybrid", or "large")
        -threshold_value: threshold for cutting off values in final output
        -bias: bias towards attention in combination step
        -combine_method: method to combine depth and attention ("sum" or "multiply")
        -scaling_factor: resolution scaling factor for combination step
        -display_dim: tuple of output dimensions (haptic display H x W)"""

        self.device = device
        self.resolution_attention = resolution_attention
        self.threshold_value = threshold_value
        self.bias = bias
        self.scaling_factor = scaling_factor
        self.display_dim = display_dim
        self.combine_method = combine_method

        # set up the MiDaS depth mode:
        depth_transforms = torch.hub.load('intel-isl/MiDaS','transforms') # predefined transforms for depth model
        if depth_model_type=='hybrid':
            self.depth_model = torch.hub.load('intel-isl/MiDaS','DPT_Hybrid')
            self.depth_transform = depth_transforms.dpt_transform
            self.gradient_size = 672
        elif depth_model_type=='large':
            self.depth_model = torch.hub.load('intel-isl/MiDaS','DPT_Large')
            self.depth_transform = depth_transforms.dpt_transform
            self.gradient_size = 672
        elif depth_model_type=='small':
            self.depth_model = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
            self.depth_transform = depth_transforms.small_transform
            self.gradient_size = 256
        self.depth_model.to(device)
        self.depth_model.eval()

        # set up the DINO attention model:
        self.attention_model = torch.hub.load('facebookresearch/dino:main','dino_vits8')
        self.attention_model.to(device)
        self.attention_model.eval()

        # set up frame objects:
        self.attention_frame = None
        self.depth_frame = None
        self.combined_frame = None
        self.threshold_frame = None
        self.output = None

    def single_run(self, frame):
        """Runs the visual-haptic processing for a single frame.
        
        Inputs:
        -frame: image of current frame grabbed from OpenCV source"""
        self.attention_frame = self.get_attention(frame)
        depth = self.get_depth(frame)
        self.depth_frame = self.remove_gradient(depth)
        self.combined_frame = self.get_combined(self.depth_frame, self.attention_frame, self.combine_method)
        self.threshold_frame = self.get_threshold(self.combined_frame)
        self.output = self.get_downsample(self.threshold_frame)
        return self.output

    # Get self attention from video frame using DINO
    def get_attention(self, frame):
        transform = transforms.Compose([           
                                    transforms.Resize((self.resolution_attention,int(np.floor(self.resolution_attention*1.7777)))),
                                    # transforms.CenterCrop(resolution), #should be multiple of model patch_size                 
                                    transforms.ToTensor(),                    
                                    #transforms.Normalize(mean=0.5, std=0.2)
                                    ])
        frame_I = Image.fromarray(frame) # convert data type
        frame_transformed = transform(frame_I).unsqueeze(0)
        attentions = self.attention_model.get_last_selfattention(frame_transformed)
        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 1:].reshape(nh,-1)
        patch_size = 4
        width_feature_map = frame_transformed.shape[-2] // patch_size
        height_feature_map = frame_transformed.shape[-1] // patch_size

        attentions = attentions.reshape(nh, width_feature_map//2, height_feature_map//2)
        attentions_interp = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=1, mode="nearest")[0].detach().numpy()
        attention_frame = (np.mean(attentions_interp, axis=0))

        return attention_frame

    # get depth estimation of frame using MiDaS
    def get_depth(self, frame):
        frame_transformed = self.depth_transform(frame) # use small image transform
        depth = self.depth_model(frame_transformed) # evaluate using small model
        depth_frame = depth.cpu().detach().numpy().squeeze(0) # formatting

        return depth_frame

    def remove_gradient(self, depth_frame):
        # remove normal depth gradient from depth map
        # depth_nm = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
        xgrid = np.zeros(self.gradient_size, dtype=float)
        ygrid = depth_frame.mean(axis=1) # make mean-depth-based gradient
        gradient_array = np.meshgrid(xgrid, ygrid)[1] # form gradient array w/ same size as depth
        depth_frame_nograd = (depth_frame - gradient_array)
        depth_frame_nograd = (depth_frame_nograd > 0) * depth_frame_nograd # take only positive values

        return depth_frame_nograd

    # combine depth and attention maps together
    def get_combined(self, depth_frame, attention_frame, method='sum'):
        depth_resized = cv2.resize(depth_frame, dsize=(16*self.scaling_factor, 9*self.scaling_factor), interpolation=cv2.INTER_CUBIC)
        depth_normal = cv2.normalize(depth_resized, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
        attention_resized = cv2.resize(attention_frame, dsize=(16*self.scaling_factor, 9*self.scaling_factor), interpolation=cv2.INTER_CUBIC)
        attention_normal = cv2.normalize(attention_resized, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)

        if method=='multiply':
            combined = depth_normal*attention_normal
        elif method=='sum':
            combined = ((1-self.bias)*depth_normal + (self.bias)*attention_normal)
        combined_frame = cv2.normalize(combined, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        return combined_frame

    # threshold the combined map to threshold_val
    def get_threshold(self, combined_frame):
        return (combined_frame > self.threshold_value) * combined_frame

    # downsample the thresholded map to the grid size of the haptic display
    def get_downsample(self, threshold_frame):
        ### new code for interpolation:
        downsampled_frame = np.zeros(self.display_dim) # initialize
        interval_H = int(np.floor(threshold_frame.shape[0]/self.display_dim[0]))
        interval_W = int(np.floor(threshold_frame.shape[1]/self.display_dim[1]))
        for rr in range(self.display_dim[0]):
            for cc in range(self.display_dim[1]):
                frame_slice = threshold_frame[rr*interval_H:(rr+1)*interval_H, cc*interval_W:(cc+1)*interval_W]
                mean_slice = np.mean(frame_slice)
                std_slice = np.std(frame_slice)
                max_slice = np.max(frame_slice)
                if mean_slice+3*std_slice > max_slice: # I'm doing some weird selection of max vs. mean depending on std of the frame slice
                    downsampled_frame[rr, cc] = max_slice
                else:
                    downsampled_frame[rr, cc] = mean_slice

        return downsampled_frame
        ### original code for interpolation:
        # return cv2.resize(thresholded, dsize=(DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA) 