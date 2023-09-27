import cv2

cap = cv2.VideoCapture('video3.mp4') #use video
ret, img = cap.retrieve()

method = cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
