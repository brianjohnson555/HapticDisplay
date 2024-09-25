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
from scipy.ndimage import median_filter

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

    def get_attention(self, frame):
        """Gets self attention from video frame using DINO self.attention_model.
        
        The frame is resized based on self.resolution_attention value."""

        transform = transforms.Compose([           
                                    transforms.Resize((self.resolution_attention,int(np.floor(self.resolution_attention*1.7777)))),
                                    # transforms.CenterCrop(resolution), #should be multiple of model patch_size                 
                                    transforms.ToTensor(),                    
                                    #transforms.Normalize(mean=0.5, std=0.2)
                                    ])
        frame_I = Image.fromarray(frame) # convert data type
        frame_transformed = transform(frame_I).unsqueeze(0)
        # get self attention from DINO:
        attentions = self.attention_model.get_last_selfattention(frame_transformed)
        # reshape attentions (from tutorial):
        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 1:].reshape(nh,-1)
        patch_size = 4
        width_feature_map = frame_transformed.shape[-2] // patch_size
        height_feature_map = frame_transformed.shape[-1] // patch_size
        attentions = attentions.reshape(nh, width_feature_map//2, height_feature_map//2)
        # interpolate and average:
        attentions_interp = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=1, mode="nearest")[0].detach().numpy()
        attention_frame = (np.mean(attentions_interp, axis=0))

        return attention_frame

    def get_depth(self, frame):
        """Get depth estimation from video frame using MiDaS self.depth_model.
        
        The depth transformation must match the model (small model=small transform)."""

        frame_transformed = self.depth_transform(frame) # image transform
        depth = self.depth_model(frame_transformed) # evaluate using matching model
        depth_frame = depth.cpu().detach().numpy().squeeze(0) # formatting

        return depth_frame

    def remove_gradient(self, depth_frame):
        """Subtracts a default depth gradient from the depth estimation.
        
        This assumes that the depth image is a typical landscape scene with foreground,
         midground, and background. The goal is to account for the closer depth of the 
         foreground and the farther depth of the background; when the gradient is subtracted,
         any parts of the frame which contrast from this gradient will have increased contrast."""

        # depth_nm = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
        xgrid = np.zeros(self.gradient_size, dtype=float)
        ygrid = depth_frame.mean(axis=1) # make mean-depth-based gradient
        gradient_array = np.meshgrid(xgrid, ygrid)[1] # form gradient array w/ same size as depth
        depth_frame_nograd = (depth_frame - gradient_array)
        depth_frame_nograd = (depth_frame_nograd > 0) * depth_frame_nograd # take only positive values

        return depth_frame_nograd

    def get_combined(self, depth_frame, attention_frame, method='sum', normalize:bool=True):
        """Combine the depth and attention mappings.
        
        Each depth/attention frame is resized and normalized before being combined. The
        'method' input specifies if the attention and depth are multiplied together or summed.
        The 'normalize' method specifies if the frame is normalized."""

        depth_resized = cv2.resize(depth_frame, dsize=(16*self.scaling_factor, 9*self.scaling_factor), interpolation=cv2.INTER_CUBIC)
        attention_resized = cv2.resize(attention_frame, dsize=(16*self.scaling_factor, 9*self.scaling_factor), interpolation=cv2.INTER_CUBIC)
        if normalize:
            depth_resized = cv2.normalize(depth_resized, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
            attention_resized = cv2.normalize(attention_resized, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)

        if method=='multiply':
            combined = depth_resized*attention_resized
        elif method=='sum':
            combined = ((1-self.bias)*depth_resized + (self.bias)*attention_resized)

        if normalize:
            combined = cv2.normalize(combined, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        return combined

    # threshold the combined map to threshold_val
    def get_threshold(self, combined_frame):
        """Thresholds the input frame based on self.threshold_value. Only values
        above the threshold will be returned; lower values become zero."""

        return (combined_frame > self.threshold_value) * combined_frame

    def get_downsample(self, threshold_frame):
        """Downsamples the input frame to the dimensions of the haptic display.
        
        I am performing a custom interpolation method to downsample the resolution. The goal
        of the method is to preserve high-valued intensities. E.g. for a given haptic pixel size,
        if there is a close-depth or high-attention item inside, we should keep its value instead
        of diluting it by average all of the values in the pixel area during the downsample.

        The original interpolation method is commented out at the end of the method."""
        
        ### new code for interpolation:
        downsampled_frame = np.zeros(self.display_dim) # initialize
        interval_H = int(np.floor(threshold_frame.shape[0]/self.display_dim[0]))
        interval_W = int(np.floor(threshold_frame.shape[1]/self.display_dim[1]))
        for rr in range(self.display_dim[0]):
            for cc in range(self.display_dim[1]):
                # look at just the pixel in question:
                frame_slice = threshold_frame[rr*interval_H:(rr+1)*interval_H, cc*interval_W:(cc+1)*interval_W]
                # compute statistics:
                mean_slice = np.mean(frame_slice)
                std_slice = np.std(frame_slice)
                max_slice = np.max(frame_slice)
                # selection:
                if mean_slice+3*std_slice > max_slice: # I'm doing some weird selection of max vs. mean depending on std of the frame slice
                    downsampled_frame[rr, cc] = max_slice # important feature detected; keep max value
                else:
                    downsampled_frame[rr, cc] = mean_slice # not important enough; use mean value

        return downsampled_frame
        ### original code for interpolation:
        # return cv2.resize(thresholded, dsize=(DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA) 


def smooth(sequence:np.ndarray, size):
    med_filt = sequence.copy()
    for row in range(sequence.shape[1]):
        for col in range(sequence.shape[2]):
            med_filt[:,row,col] = median_filter(sequence[:,row,col], 
                                            size=size, 
                                            mode='reflect', 
                                            cval=0.0, 
                                            origin=0, 
                                            axes=None)
    return med_filt