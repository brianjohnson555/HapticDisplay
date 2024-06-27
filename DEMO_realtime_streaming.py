###### USER SETTINGS ######
SERIAL_ACTIVE = True
SAVE_VIDEO = False
NUM_SWITCHBOARDS = 2
COM1 = "COM9" #bottom
COM2 = "COM12" #top

###### INITIALIZATIONS ######
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import serial

###### MAIN ######
# Set up pixel to switchboard mapping:
# bot: |2|4|6|8|10|
# bot: |1|3|5|7|9|
# top: |2|4|6|8|10|
# top: |1|3|5|7|9|

def map_pixels(output):
    periods_bot = np.array([output[1,0], output[0,0], output[1,1], output[0,1], output[1,2], output[0,2], output[1,3], output[0,3], output[1,4], output[0,4]])
    periods_top = np.array([output[3,0], output[2,0], output[3,1], output[2,1], output[3,2], output[2,2], output[3,3], output[2,3], output[3,4], output[2,4]])
    return periods_bot, periods_top


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
        packet = b''.join(packetlist) # add space
        return packet
    
    if NUM_SWITCHBOARDS > 1:
        ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS),
                serial.Serial(COM2, 9600, timeout=0, bytesize=serial.EIGHTBITS)]
    else:
        ser = [serial.Serial(COM1, 9600, timeout=0, bytesize=serial.EIGHTBITS)]

    for switchboard in ser:
        switchboard.write('E'.encode()) # Enable HV
        time.sleep(0.1)

if SAVE_VIDEO:
    outlist = []
    outlist2 = []
    
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

    # resize and threshold
    depth_re = cv2.resize(depth_sub, dsize=(1*5, 1*4), interpolation=cv2.INTER_CUBIC)
    depth_nm = cv2.normalize(depth_re, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    threshold = 0.25
    output = (depth_nm > threshold) * depth_nm

    ## LINEAR MAPPING FROM INTENSITY TO FREQUENCY (TO DISPLAY)
    mapped_freq = 20*output # mapped frequency (Hz)
    mapped_freq[mapped_freq==0] = 0.01
    mapped_per = np.reciprocal(mapped_freq) # mapped period (sec)
    mapped_per_ms = 1000*mapped_per # mapped period (ms)
    mapped_per_ms = mapped_per_ms.astype(int)
    mapped_per_ms[mapped_per_ms>500] = 0 # anything above 500 ms = 0 (below 2 Hz = 0)


    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = img
        continue
    
    # Show frames
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Video',img)
    # cv2.namedWindow('Output',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('Output',output)
    if SAVE_VIDEO:
        outlist.append(imgcolor)
        outlist2.append(output)

    if SERIAL_ACTIVE:
        if NUM_SWITCHBOARDS>1:
            
            periods_bot, periods_top = map_pixels(mapped_per_ms)
            packet_bot = make_packet(periods_bot)
            packet_top = make_packet(periods_top)
            ser[0].write(packet_bot)
            ser[1].write(packet_top)
        else:
            period = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            packet = make_packet()
            ser[0].write(packet)
    
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break # BREAK OUT OF LOOP WHEN "B" KEY IS PRESSED!


if SAVE_VIDEO:
    import matplotlib.animation as animation
    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist)):
        im = plt.imshow(outlist[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "camera_output.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=10)
    plt.close()

    ims = []
    figure = plt.figure()
    for i in range(0,len(outlist2)):
        im = plt.imshow(outlist2[i], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(figure, ims, blit=True, repeat=False)
    filename = "algo_output.mp4"
    ani.save(filename, writer = "ffmpeg", bitrate=1000, fps=10)
    plt.close()

if SERIAL_ACTIVE:
    for switchboard in ser:
        switchboard.write('D'.encode()) # Disable HV
    time.sleep(0.1)  
    for switchboard in ser:
        switchboard.close() # close serial port