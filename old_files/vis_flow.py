import numpy as np
import cv2 as cv

cap = cv.VideoCapture('video3.mp4')
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
prvs_r = cv.resize(prvs, (160, 90))
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    next_r = cv.resize(next, (160, 90))

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, pyr_scale = 0.5, levels = 5, winsize = 5, iterations = 10, poly_n = 7, poly_sigma = 1.5, flags = 0)
    diff = cv.absdiff(prvs_r, next_r)
    
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', diff)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs_r = next_r
cv.destroyAllWindows()