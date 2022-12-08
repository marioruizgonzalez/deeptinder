import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from flask import Flask, jsonify, request, make_response

import argparse
import uuid
import json
import time
from tqdm import tqdm

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

#------------------------------

from deepface import DeepFace

import cv2
import matplotlib.pyplot as plt
import pandas as pd

df = DeepFace.find(img_path="kpop-asiachan.jpg", db_path="./park/")

df.head()