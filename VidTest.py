import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while (1):
    _, frame = cap.read()
    cv2.imshow('frame', frame)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
