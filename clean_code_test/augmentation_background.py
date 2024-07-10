# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:31:11 2023

@author: DELL
"""

import sys
import os
import cv2
import argparse
from rembg import remove
import numpy as np
import matplotlib.pyplot as plt
import random
import imutils

def get_cloth_without_bg(img_cloth, mask_cloth):

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(mask_cloth, kernel)
    mask_cloth = cv2.threshold(erode, 50, 255, cv2.THRESH_BINARY)[1] // 255
    
    # create mask with same dimensions as image
    img_cloth_without_bg = np.zeros_like(img_cloth)
    for i in range(3):
        img_cloth_without_bg[:,:,i] = img_cloth[:,:,i] * mask_cloth
    return img_cloth_without_bg

def get_cntr(mask_cloth):
    # Get contour from prediction 
    mask = mask_cloth.astype(np.uint8)
    cntrs, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntr = max(cntrs, key=cv2.contourArea)
            
    # Transform contour data in polygon 
    polygon = []
    for i in range(len(cntr)):
        x, y = cntr[i][0][0], cntr[i][0][1]
        polygon.append([x, y])       
    return np.array(polygon)

def get_bbx(cntr):
    
    # get bounding box of the mask 
    x1, y1 = np.min(np.ravel(cntr)[0::2]), np.min(np.ravel(cntr)[1::2])
    x2, y2 = np.max(np.ravel(cntr)[0::2]), np.max(np.ravel(cntr)[1::2])  
    
    # Define bounding box
    bbx = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
    return bbx

def set_light(img_cloth):
    return (np.random.randint(0,img_cloth.shape[1]), np.random.randint(0,img_cloth.shape[0]))

def get_shadow(bbx, lux, cntr, img_cloth):
    
    # Store img dimensions
    h, w, c = img_cloth.shape
    
    # Compute middle of the bounding box 
    x1, y1, x2, y2 = bbx[0][0], bbx[0][1], bbx[2][0], bbx[2][1]
    mx, my = np.abs(x2-x1)//2, np.abs(y2-y1)//2
    
    # Compute euclidean distances from middle bbx to each corner of image
    d1 = np.sqrt((mx-0)**2 + (my-0)**2) # to top-left corner
    d2 = np.sqrt((mx-w)**2 + (my-0)**2) # to top-right corner
    d3 = np.sqrt((mx-w)**2 + (my-h)**2) # to down-right corner
    d4 = np.sqrt((mx-0)**2 + (my-h)**2) # to down-left corner

    # Intensitiy of the deformation
    # Arbitrary set : max distance <=> 100 displacement pixels 
    gamma = 100 * np.sqrt((mx-lux[0])**2 + (my-lux[1])**2) /  max(d1, d2, d3, d4)

    dx, dy = int(gamma * np.sign(mx-lux[0])), int(gamma * np.sign(my-lux[1]))
        
    # Définir les points de destination pour la transformation de perspective
    bbx2 = np.array([[x1+dx,y1+dy], [x2+dx,y1+dy], [x2+dx,y2+dy], [x1+dx,y2+dy]])

    # Créer une matrice de transformation de perspective
    matrice = cv2.getPerspectiveTransform(bbx.astype(np.float32), bbx2.astype(np.float32))

    # Transformer le polygone2 en utilisant la matrice de transformation de perspective
    res = cv2.perspectiveTransform(cntr.reshape(-1, 1, 2).astype(np.float32), matrice)

    shadow = []
    for i in range(len(res)):
        x, y = int(res[i][0][0]), int(res[i][0][1])
        shadow.append([x, y])
        
    return np.array(shadow)  

def paste2back(bg, fg):
    
    fgh, fgw = fg.shape[:-1]
    bgh, bgw = bg.shape[:-1]
    
    scale = min(bgh/fgh, bgw/fgw)

    tempw, temph = int(scale*fgw), int(scale*fgh)
    dim = (tempw, temph)
    new = cv2.resize(fg, dim, interpolation=cv2.INTER_AREA)
    
    # introduire un aspect aléatoire dans les offsets
    offx, offy = 0, 0
    
    if (bg.shape[1] - new.shape[1] > 0):
        offx = np.random.randint(0, bg.shape[1] - new.shape[1])
        
    if (bg.shape[0] - new.shape[0] > 0):
        offy = np.random.randint(0, bg.shape[0] - new.shape[0])
    
    # define region of interest (roi) area 
    x_min, x_max = offx, new.shape[1] + offx
    y_min, y_max = offy, new.shape[0] + offy
        
    bg_roi = bg[y_min:y_max, x_min:x_max]
    gray_fg = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(gray_fg, 0, 255, cv2.THRESH_BINARY_INV)
    bg_hole = cv2.bitwise_and(bg_roi, bg_roi, mask=mask)
    blending = cv2.add(bg_hole, new)
    bg[y_min:y_max, x_min:x_max] = blending

    return bg
    
 
def add_shadow_and_background(cloth_with_stained_without_bg, mask_cloth, img_background):
    # Compute external contour
    cntr = get_cntr(mask_cloth)
    
    # Calculate bounding box 
    bbx  = get_bbx(cntr)
    
    # Set lux position 
    lux = set_light(cloth_with_stained_without_bg)#, windowName='Lux location')
    
    # Calculate shadow cntr based on lux position
    shadow = get_shadow(bbx, lux, cntr, cloth_with_stained_without_bg)
    
    # Afficher les points du polygone transformé
    mask = np.zeros(cloth_with_stained_without_bg.shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, pts=[shadow], color=(50,50,50))
    # mask = cv2.blur(mask,ksize=(1000,1000))
    
    # First add shadow img without back on shadow
    temp = paste2back(mask, cloth_with_stained_without_bg)
    
    # Then add the whole thing to your background
    pic = paste2back(img_background, temp)
    return pic

### 



