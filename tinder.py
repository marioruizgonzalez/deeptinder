
import numpy as np
import cv2
from PIL import ImageGrab
import dlib
import keyboard
import os
import pyautogui
import time
from helper import is_tinder_open, press_dislike, press_like

THRESH = 10.0
SCREEN_H, SCREEN_W = 1366, 768
CAP_H, CAP_W = SCREEN_H//2, SCREEN_W//2
FACE_H, FACE_W = 224, 224
padding = 50

detector = dlib.get_frontal_face_detector()

while(True):
    if True:

        screen_cap = ImageGrab.grab(bbox=(0, 0, CAP_H, CAP_W))
        screen_cap_num = np.array(screen_cap)
        screen_cap = cv2.cvtColor(screen_cap_num, cv2.COLOR_BGR2RGB)

        screen_cap_copy = screen_cap.copy()

        faces = detector(screen_cap)
        print('llego aqui')
        if faces:
            face = faces[0]
            x, y, x1, y1 = face.left(), face.top(), face.right() + \
                padding, face.bottom()+padding
            cv2.rectangle(screen_cap, (x, y), (x1, y1), (0, 255, 0), 2)

            face_img = screen_cap_copy[y:y1, x:x1]
            face_img = cv2.resize(face_img, (FACE_H, FACE_W))
            score = 9.0 #pred(face_img)

            if score > THRESH:

                press_like()
            elif score <= THRESH:
                press_dislike()

            cv2.imshow("Face", face_img)

        else:
            press_dislike()

        cv2.imshow("Screen Capture", screen_cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('No hiso nada')
        cv2.destroyWindow("Screen Capture")
        continue

cv2.destroyAllWindows()
