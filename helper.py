import pyautogui
import time
import numpy as np


dislike_path = "buttons/dislike.png"
like_path = "buttons/like.png"
t_path = "buttons/t.png"


def is_tinder_open():
    print('Entro is_tinder_open')
    if pyautogui.locateOnScreen(t_path, grayscale=True, confidence=.5):
        print('Si lo encontro el banner')
        return True
    else:
        False


def press_like():
    print('Entro press_like')
    if pyautogui.locateOnScreen(like_path, grayscale=True, confidence=.75):

        time.sleep(10)
        print("Like")
        print(pyautogui.locateOnScreen(
            like_path, grayscale=True, confidence=.75))
        pyautogui.click(pyautogui.locateOnScreen(
            like_path, grayscale=True, confidence=.75))


def press_dislike():
    print('Entro press_dislike')
    if pyautogui.locateOnScreen(dislike_path, grayscale=True, confidence=.75):

        time.sleep(10)
        print("Dislike")
        print(pyautogui.locateOnScreen(
            dislike_path, grayscale=True, confidence=.75))
        pyautogui.click(pyautogui.locateOnScreen(
            dislike_path, grayscale=True, confidence=.75))
        
def identifica():
    res = pyautogui.locateOnScreen(dislike_path)
    print(res)
    edit_but = pyautogui.center(res)
    pyautogui.moveTo(edit_but)


identifica()