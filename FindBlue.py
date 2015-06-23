import cv2
import numpy as np

def find_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([50, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blur = cv2.inRange(cv2.GaussianBlur(mask, (5,5), 0), 250, 255)
    im2, contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_cnt = None
    largest_cnt_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (largest_cnt == None or area > largest_cnt_area):
            largest_cnt = cnt
            largest_cnt_area = area

    x,y,w,h = cv2.boundingRect(largest_cnt)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.circle(img, (x+w/2, y+h/2), 3, (0,0,255), -1)
    
    return(x+w/2, y+h/2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while (1):
        _, frame = cap.read()
        x,y = find_blue(frame)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
