import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Load MiDaS model onto CPU
device = torch.device('cpu')
midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Setup frame capture
cap = cv2.VideoCapture('video2.mp4') #use video
# cap = cv2.VideoCapture(0) #stream from webcam
previous_frame = None
    
while True: 
    # Load frame
    ret,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(img,1.5,img,0,1)

    # Compute depth
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(120, 160),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    depth = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    #depth = (depth).astype(np.uint8)

    if (previous_frame is None):
        # First frame; there is no previous one yet
        previous_frame = img
        continue
    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=img)
    previous_frame = img
    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)
    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=100, maxval=255, type=cv2.THRESH_BINARY)[1]
    im = cv2.cvtColor(thresh_frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image=im, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    mini = cv2.resize(depth, (6,4), interpolation = cv2.INTER_AREA)
    # Show frames
    cv2.namedWindow('Video',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Video',img)
    # cv2.namedWindow('Contrast',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('Contrast',img2)
    cv2.namedWindow('Mini',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Mini',depth)
    #cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #cv2.imshow('thresh',thresh_frame)
    
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break