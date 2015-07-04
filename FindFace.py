import cv2
import numpy as np

def find_face(img):
    face_cascade = cv2.CascadeClassifier('c:/Users/tzadikar/git/cv/haarcascade_frontalcatface.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    (xmax, ymax, wmax, hmax, amax) = (-1, -1, -1, -1, -1)
    for (x,y,w,h) in faces:
        area = w*h
        if area > amax:
            (xmax, ymax, wmax, hmax, amax) = (x,y,w,h, area)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.circle(img, (xmax+wmax/2, ymax+hmax/2), 3, (0,0,255), -1)
    
    return(xmax+wmax/2, ymax+hmax/2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while (1):
        _, frame = cap.read()
        x,y = find_face(frame)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
