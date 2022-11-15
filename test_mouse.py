import pyautogui
from pynput.mouse import Button, Controller
import time

def dislike():
    mouse = Controller()
    pyautogui.moveTo(200,750)
    mouse.click(Button.left)
    mouse.click(Button.left)
    
def like():
    mouse = Controller()
    pyautogui.moveTo(450,750)
    mouse.click(Button.left)
    mouse.click(Button.left)

like()

