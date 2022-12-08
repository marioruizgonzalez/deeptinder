from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import pandas as pd

df = DeepFace.find(img_path="kpop-asiachan.jpg", db_path="./park/")

df.head()