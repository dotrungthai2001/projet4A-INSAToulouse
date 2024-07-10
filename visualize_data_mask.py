# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:55:26 2023

@author: TranK
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import matplotlib.pyplot as plt
import numpy as np
from parameters import *

# path_base = "C:/Users/Asus/Pictures/dataset/" #os.getcwd()

# cloth_folder = "dataset_2_10img/"
# stain_folder = path_base +"stain/"

path_test = path_base + cloth_folder

paste_to = path_base +"DATA/"
if not os.path.exists(paste_to):
    os.mkdir(paste_to)
    
annotation = path_base +"MASK/"
if not os.path.exists(annotation):
    os.mkdir(annotation)
    
DATA_path = [image_path.replace("\\", "/") for image_path in glob.glob(paste_to + "*")]

MASK_path = [image_path.replace("\\", "/") for image_path in glob.glob(annotation + "*")]

print("Number of input :", len(DATA_path))
print("Number of mask :", len(MASK_path))

for i in range(10):
    img_input = cv2.imread(DATA_path[i])
    img_mask = cv2.imread(MASK_path[i])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)) #row=0, col=0
    ax[1].imshow(cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)) #row=0, col=0
