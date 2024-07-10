# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:43:11 2023

@author: TranK

This file analyse RGB histogram, grayscale histogram of stain image, 
analyse texture by graylevel cooccurence matrix and label binary patter of stain image.
"""


import matplotlib.pyplot as plt
import os 
import glob
import numpy as np
import cv2
import skimage
from parameters import *

stain_test = glob.glob(stain_folder+"*")
stain_raw_images = [image_path.replace("\\", "/") for image_path in stain_test]

for stain_path in stain_raw_images[:10]: 
    img_stain = cv2.imread(stain_path)
    colors = ("red", "green", "blue")

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(cv2.cvtColor(img_stain, cv2.COLOR_BGR2RGB)) #row=0, col=0

    ax[1].set_xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            img_stain[:, :, channel_id], bins=256, range=(0, 256)
        )
        ax[1].plot(bin_edges[0:-1], histogram, color=color)
    ax[1].set_title("Color Histogram")
    ax[1].set_xlabel("Color value")
    ax[1].set_ylabel("Pixel count")
    
    
################################
for stain_path in stain_raw_images[:10]: 
    img_stain = cv2.imread(stain_path)
    gray = cv2.cvtColor(img_stain, cv2.COLOR_BGR2GRAY)
    histogram, bin_edges = np.histogram(gray/255., bins=256, range=(0, 1))

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(gray, cmap='gray')
    ax[1].set_title("Grayscale Histogram")
    ax[1].set_xlabel("Color value")
    ax[1].set_ylabel("Pixel count")
    ax[1].set_xlim([0.0, 1.0])  # <- named arguments do not work here

    ax[1].plot(bin_edges[0:-1], histogram)  # <- or here



#######################
# GLCM gray level cooccurence matrix
for stain_path in stain_raw_images[:10]: 
    print("Image : ", stain_path)
    img_stain = cv2.imread(stain_path)
    gray = cv2.cvtColor(img_stain, cv2.COLOR_BGR2GRAY)
    
    # Find the GLCM
    
    # Param:
    # source image
    # List of pixel pair distance offsets - here 1 in each direction
    # List of pixel pair angles in radians
    graycom = skimage.feature.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    
    # Find the GLCM properties
    contrast = skimage.feature.greycoprops(graycom, 'contrast')
    dissimilarity = skimage.feature.greycoprops(graycom, 'dissimilarity')
    homogeneity = skimage.feature.greycoprops(graycom, 'homogeneity')
    energy = skimage.feature.greycoprops(graycom, 'energy')
    correlation = skimage.feature.greycoprops(graycom, 'correlation')
    ASM = skimage.feature.greycoprops(graycom, 'ASM')
    
    print("Contrast: {}".format(contrast))
    print("Dissimilarity: {}".format(dissimilarity))
    print("Homogeneity: {}".format(homogeneity))
    print("Energy: {}".format(energy))
    print("Correlation: {}".format(correlation))
    print("ASM: {}".format(ASM))
    

#######################
# Local binary pattern


class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    lbp = skimage.feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp

for stain_path in stain_raw_images: 
    img_stain = cv2.imread(stain_path)

    gray = cv2.cvtColor(img_stain, cv2.COLOR_BGR2GRAY)
    desc = LocalBinaryPatterns(24, 8)
    hist, lbp = desc.describe(gray)
    # print("Histogram of Local Binary Pattern value: {}".format(hist))
    
    # contrast = contrast.flatten()
    # dissimilarity = dissimilarity.flatten()
    # homogeneity = homogeneity.flatten()
    # energy = energy.flatten()
    # correlation = correlation.flatten()
    # ASM = ASM.flatten()
    # hist = hist.flatten()
    
    # features = np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation, ASM, hist), axis=0) 
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(cv2.cvtColor(img_stain, cv2.COLOR_BGR2RGB)) #row=0, col=0

    ax[1].imshow(lbp)
