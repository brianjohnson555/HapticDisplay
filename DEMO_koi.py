###### USER SETTINGS ######
FILENAME = "algo_input_data/datakoi_output.txt"
VIDEONAME = "algo_input_videos/video_koi.mp4"
COM_A = "COM9"
COM_B = "COM15"
COM_C = "COM16"
SERIAL_ACTIVE = True

###### INITIALIZATIONS ######
import cv2
import time
import serial
import utils.algo_functions as algo_functions # my custom file
from numpy import genfromtxt

###### MAIN ######
# Load data
data=genfromtxt(FILENAME,delimiter=',')[:,0:28]
data_length = data.shape[0]

# Set up USBWriter:
serial_ports = [COM_A, COM_B, COM_C]
USB_writer = algo_functions.USBWriter(serial_ports, serial_active=SERIAL_ACTIVE)

# Enable HV!!!
USB_writer.HV_enable()

while True:
    isClose=False # break condition
    cap = cv2.VideoCapture(VIDEONAME) # set up video capture
    dataindex = 0 # initialize data start point

    while True:
        ret,img = cap.read() # read frame from video

        if ret and dataindex<data_length: # if frame exists, run; otherwise, video is finished->loop back to beginning
            intensity_array = data[dataindex,:] # read each line of data
            dataindex += 1 # update data index
            haptic_output = algo_functions.map_intensity(intensity_array) # map from algo intensity to duty cycle/period
            USB_writer.write_to_USB(haptic_output)

            # Display video:
            cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video',cv2.flip(img, -1))
            # time.sleep(0.05)

            if(cv2.waitKey(10) & 0xFF == ord('b')): # if user pressed 'b' for break
                isClose = True # assign stop flag
                break
        else: 
            break # video is finished, break and reset frame count

    if isClose: 
        break # user pressed 'b', stop script
    
# Disable HV!!!
USB_writer.HV_disable()
time.sleep(0.5)
