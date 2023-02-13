import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

device = torch.device('cpu')
midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small")

midas.to(device)
midas.eval()


filename = "input.jpg"

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
    
while True: 
    
    ret,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    depth = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_64F)
    depth = (depth*255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)



    cv2.imshow('Video', depth)
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break