from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import pandas as pd
""" 
def verify(img1_path, imag2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(imag2_path)

    plt.imshow(img1[:,:,::-1])
    plt.show()
    plt.imshow(img2[:,:,::-1])
    plt.show() 
  
    output= DeepFace.verify(img1_path, imag2_path)

    print(output)

    verification = output['verified']
    if verification:
        print('verified')
    else:
        print('No Valid')
    
verify('eun-bin1.jpg','eun-bin1.jpg')
"""