# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:47:04 2023

@author: DELL
"""

list_area_path = 'C:/Users/DELL/Documents/PROJET/dataset/information_dataset_folder/info_fulldata_alpha0.25-1_bg+shadow/dict_area.pkl'
list_color_path = "C:/Users/DELL/Documents/PROJET/dataset/information_dataset_folder/info_fulldata_alpha0.25-1_bg+shadow/dict_color.pkl"
%matplotlib qt
import matplotlib.pyplot as plt
import pickle
import numpy as np
from parameters import *


def stats_color(dict_color_path, file_images): 
    '''
    file_images : labeled_training_images or labeled_validation_images file, contain images name (?)
    '''
    with open(dict_color_path,"rb") as f:
        dict_color = pickle.load(f)

    list_color =[]
    for key in list_color.keys():
        if key in file_images:
           for value in list_color[key]:
               list_color.append(value)
    list_color = np.array(list_color)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # r = list_color[:,0]
    # g = list_color[:,1]
    # b = list_color[:,2]
    # ax.scatter(r,g,b,s=5, facecolors=list_color/255.)
    
    # ax.set_xlabel('R Label')
    # ax.set_ylabel('G Label')
    # ax.set_zlabel('B Label')
    
    # plt.show()
    # # plt.savefig('color_plot.jpg')
    return list_color

###################
## list_area.plk contains a list of floats (0 < stain surface/cloth surface <1)   
def stats_area(dict_area_path, file_images):
    with open(dict_area_path,'rb') as f:
        dict_area = pickle.load(f)
    
    list_area =[]
    for key in list_area.keys():
        if key in file_images:
           for value in list_area[key]:
               list_area.append(value)
    list_area = np.array(list_area)
    
    # plt.hist(list_area, bins=30)
    # plt.show()
    return list_area

    
    
    
    
    
    
    
    
    
    
    
    
    
    