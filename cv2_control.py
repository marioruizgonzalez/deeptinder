import numpy as np
import cv2
from PIL import ImageGrab
import dlib
import time

THRESH = 10.0
CAP_H, CAP_W = 720, 768
FACE_H, FACE_W = 224, 224
padding = 50

detector = dlib.get_frontal_face_detector()
screen_cap = ImageGrab.grab(bbox=(0, 0, CAP_H, CAP_W))
screen_cap_num = np.array(screen_cap)
screen_cap = cv2.cvtColor(screen_cap_num, cv2.COLOR_BGR2RGB)

screen_cap_copy = screen_cap.copy()

faces = detector(screen_cap)


cv2.imshow('Logo OpenCV',screen_cap)
cv2.waitKey(0)
cv2.destroyAllWindows()


face = faces[0]
x, y, x1, y1 = face.left(), face.top(), face.right() + \
padding, face.bottom()+padding
cv2.rectangle(screen_cap, (x, y), (x1, y1), (0, 255, 0), 2)

face_img = screen_cap_copy[y:y1, x:x1]
face_img = cv2.resize(face_img, (FACE_H, FACE_W))

cv2.imshow('Logo OpenCV',face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()