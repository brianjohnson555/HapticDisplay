import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial
### USER SETTINGS
SERIAL_ACTIVE = False
NUM_SWITCHBOARDS = 2
COM1 = "COM9"
COM2 = "COM12"

# Load MiDaS model onto CPU
device = torch.device('cpu')
midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Setup frame capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #stream from webcam
previous_frame = None

# Setup serial connection
if SERIAL_ACTIVE: 
    
    def make_packet(periods): # Build a packet of period data to send to USB
        packetlist = []
        packetlist.append(('P').encode()) # encode start of period array
        for period in periods:
            packetlist.append((period.item()).to_bytes(2, byteorder='little')) # convert to 16bit
        packet = b''.join(packetlist) #
        return packet
    
    if NUM_SWITCHBOARDS > 1:
        ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS),
                serial.Serial(COM2, 9600, timeout=0, bytesize=serial.EIGHTBITS)]
    else:
        ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS)]

    for switchboard in ser:
        switchboard.write('E'.encode()) # Enable HV
        time.sleep(0.1)
    
while True: 
    # Load frame
    ret,img = cap.read()
    imgcolor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgcolor = cv2.addWeighted(imgcolor,1.5,imgcolor,0,1)

    # Compute depth
    frame = midas_transforms.small_transform(imgcolor) # use small image transform
    depth_size_x = 256
    depth = midas(frame) # evaluate using small model
    depth = depth.cpu().detach().numpy().squeeze(0)
    # remove normal depth gradient from depth map
    depth_nm = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    xgrid = np.zeros(depth_size_x, dtype=float)
    ygrid = depth_nm.mean(axis=1) # make mean-depth-based gradient
    grad_array = np.meshgrid(xgrid, ygrid)[1] # form gradient array w/ same size as depth
    depth_sub = (depth_nm - grad_array)
    depth_sub = (depth_sub > 0) * depth_sub # take only positive values
    depth_re = cv2.resize(depth_sub, dsize=(7, 4), interpolation=cv2.INTER_CUBIC)
    depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    threshold = 0.25
    output = (depth_nm > threshold) * depth_nm

    print(output.shape)

    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = img
        continue
    
    # Show frames
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Video',img)
    cv2.namedWindow('Output',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Output',output)

    periods = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0])

    if SERIAL_ACTIVE:
        if NUM_SWITCHBOARDS>1:
            period1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            period2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            packet1 = make_packet(period1)
            packet2 = make_packet(period2)
            ser[0].write(packet1)
            ser[1].write(packet2)
        else:
            period
            packet = make_packet()
            ser[0].write(packet)
    
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break

if SERIAL_ACTIVE:
    ser.write('D'.encode()) # disable HV
    time.sleep(0.1)  
    ser.close()