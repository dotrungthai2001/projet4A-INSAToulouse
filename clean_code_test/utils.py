import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from rembg import remove
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import binom
from helper_functions import *
from pathlib import Path
import yaml
import random
import shutil
from skimage.measure import label, regionprops
import time 
from opensimplex import OpenSimplex

FILE = Path(__file__).resolve()
PASTE_TO = FILE.parents[0].parents[0]  # folder parent of project containing code

random.seed(0)

def parse(known=False):
    '''
    Get command arguments lines
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', '--clothes_folder', type=str, required=True, help="path to cloth only dataset")
    parser.add_argument('-bg', '--backs_folder', type=str, default=None, help="path to images of background for augmentation")
    parser.add_argument('--paste_to_folder', type=str, default=PASTE_TO, help="path to directory to generate new images, default is the parent of code directory")
    parser.add_argument('--algo_gen',nargs='+', default=["CircleV1", "CircleV2", "EdgeRad", "OpenSimplexNoise"],
                        help='create sythetic data with algo CircleV2, CircleV3 or EdgeRad or OpenSimplexNoise or Random')
    parser.add_argument('--subset_use', type=int, default=None, help='Number of images taken from original data as subset to run code')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def list_full_paths(folder):
    '''
    Get path of all files in a folder, rewrite path if window slash
    '''
    path_list = glob.glob(folder + "*")
    return [path_list.replace("\\", "/") for path_list in path_list]

def compute_ratio_number_pixels(mask_stain, h_img, w_img):
    '''
    Compute ratio of area of stain over the area of images of clothes
    '''
    #area of stain is computed as number of not null pixels in mask_stain
    gray = cv2.cvtColor(mask_stain, cv2.COLOR_RGB2GRAY)
    return float(cv2.countNonZero(gray))/float(h_img*w_img)

def generate_random_RGBcolor():
    '''
    Generate list of 3 random values for rgb color
    '''
    r = np.random.randint(0,255)
    g = np.random.randint(0,255)
    b = np.random.randint(0,255)
    return [r,g,b]

def get_close_bbox_of_coords(x_list, y_list):
    '''
    Get close bounding box coordinates from x-coordinates and y-coordinates
    '''
    return np.ceil(np.array([min(x_list), min(y_list), max(x_list)-min(x_list), max(y_list)-min(y_list)])).astype(int)

def convert_SkimageToYoloFormat(list_bb_skimage, h_image, w_image):
    '''
    Convert box coordinates format from Skimage to Yolo format 
    '''
    h_image, w_image = float(h_image), float(w_image)
    list_bb_yolo = []
    for box in list_bb_skimage:
        y_min, x_min, y_max, x_max = [i for i in box]
        w_box, h_box = x_max - x_min, y_max - y_min
        x_center, y_center = (x_min+w_box/2., y_min+h_box/2.)
        list_bb_yolo.append([x_center/w_image, y_center /
                             h_image, w_box/w_image, h_box/h_image])
    return list_bb_yolo

def normalized_bbox_YoloFormat(bbox, h_image, w_image):
    '''
    Normalize boundingbox coordinates by height and width of image for yolo format
    '''
    h_image, w_image = float(h_image), float(w_image)
    x_center, y_center, w_box, h_box = [float(bbox[i]) for i in range(4)]
    return [x_center/w_image, y_center/h_image, w_box/w_image, h_box/h_image]

def distort_random_axis_shape(xi, yi, min_distort_coef=1.,max_distort_coef=4.):
    '''
    Make distortion on random 2D axis to 2D coordinates
    '''
    distort_coef = np.random.uniform(min_distort_coef,max_distort_coef)
    distort_angle = np.random.uniform()*2*np.pi
    yi /= distort_coef
    new_xi = xi*np.cos(distort_angle) + yi*np.sin(distort_angle)
    new_yi = -xi*np.sin(distort_angle) + yi*np.cos(distort_angle)
    return new_xi, new_yi

def get_mask3channels_stainraw(stain_gray):
    '''
    Get binary mask 3 channels from grayscale image
    '''
    (thresh, mask) = cv2.threshold(stain_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    try: 
        assert len(np.unique(mask)) == 2 #0 and 255 only 
    except:
        print(np.unique(mask))
        plt.imshow(stain_gray)
        raise Exception()
    invert_stain = cv2.bitwise_not(mask)
    invert_3ch = cv2.merge([invert_stain, invert_stain, invert_stain])
    
    return invert_3ch

def generate_deform_circle(n, num_spike, hyp):
    
    theta = 2 * np.pi * np.linspace(0, 1,n)
    
    if hyp.get('periodic_spike') == True: #periodic creation of spike around the circle 
        period = 2*n//num_spike

        index_spike = np.array([i*(period-1) for i in range((n-2)//(period-1)+1)])
        index_neighbor = index_spike + 1
        
        index_spike = np.concatenate([index_spike, index_neighbor])
        index_spike = np.unique(index_spike)

        not_spike = np.array([i for i in range(n-2) if i not in index_spike]) 
                
    else: #not periodic, generate random spike
        
        index_spike = random_points1D_withmindst(low_bound=1, up_bound=n-3, mindst=3, npoints=num_spike//2)
        index_neighbor = index_spike + 1
        
        index_spike = np.concatenate([index_spike, index_neighbor])
        index_spike = np.unique(index_spike)
        
        not_spike = np.array([i for i in range(n-2) if i not in index_spike])
    
    random_noise = np.random.uniform(low=hyp.get('min_spike_coef'), high=hyp.get('max_spike_coef'), size=(n-2,2))
    random_noise[not_spike] =  np.random.uniform(low=hyp.get('min_nospike_noise'), high=hyp.get('max_nospike_noise'), size=(len(not_spike), 2))
        
    y = np.c_[np.cos(theta), np.sin(theta)]
    y[1:-1, :] = np.c_[np.cos(theta), np.sin(theta)][1:-1, :] * random_noise
    
    t = np.arange(len(y[:, 0]))
    ti = np.linspace(0, t.max(), 10 * t.size)
    xi = interp1d(t, y[:, 0], kind='cubic')(ti)
    yi = interp1d(t, y[:, 1], kind='cubic')(ti)
    
    return xi, yi

def generate_random_shape(n):
    a = generate_randompoints2D_withmindistance(n, scaling_factor=9)
    rad = np.random.choice([0.2, 0.3, 0.4])

    xi, yi = interpolate_bezier(a, radius_coef=rad, angle_combine_factor=0.5)[:2]
    return xi, yi

def generate_circle_opensimplexnoise(roi_size, ratio_box_stain = 3, noise_level=50):
    
    #center in middle for non smooth cut at border of stain
    #the image should be square with shape img_size x img_size
    assert ratio_box_stain >= 2. #if not all the square roi will be stain, thus square stain...
    roi_center = roi_size // 2 
    radius = roi_size // ratio_box_stain # the "somewhat" radius of the final stain
    
    # meshgrid of pixel 
    x_coordinates = np.arange(roi_size)
    y_coordinates = np.arange(roi_size)
    xx, yy = np.meshgrid(x_coordinates, y_coordinates)
    # center the meshgrid to the roi center
    xx -= roi_center
    yy -= roi_center
    # array for distances from center
    distances = np.sqrt(xx**2 + yy**2)
    
    # Initialize noise method with seed varied per milli seconds...
    seed = int(time.time()*10**3)
    noise = OpenSimplex(seed=seed).noise2
    
    # create noise array with small numbers and scale it to range [-1, 1]
    # the scaling factor here = 50 is defined by testing... 
    noise_array = np.array([[noise(x/50, y/50) for x in x_coordinates] for y in y_coordinates])
    noise_array = noise_array / np.max(np.abs(noise_array))
    
    # deform distances array 
    distances_with_noise = distances + noise_array*noise_level
    
    # Create signed distance field : - inside object, + outside object
    sdf = distances_with_noise - radius
    
    # generate mask of stains from sdf
    mask = (sdf < 0)*255
    return mask


def insert_stain_raw_in_img(img_cloth, img_stain, mask3ch_stain, mask_cloth, mask_all_stain):
    '''
    insert stain image at a random position on the cloth image
    '''
    assert img_cloth.shape == mask_all_stain.shape

    cloth_with_stained = np.copy(img_cloth)
    h_cloth, w_cloth = cloth_with_stained.shape[:2]
    
    #while loop (current code) to avoid cases where inserting stain at negative pixel position
    count = 0
    while 1: 
        try : 
            #select random pixel in the clothes
            roi_x, roi_y = np.where(mask_cloth==255.)
            random_coord = np.random.randint(len(roi_x))
            
            #start_x, start_y is the center of the stain image
            start_x, start_y = roi_x[random_coord], roi_y[random_coord]
            h_stain, w_stain = img_stain.shape[:2]
            
            #save bounding box yoloformat for labels
            #remark that we change y to x and x to y
            #copy region of interest fo clothes to roi 
            roi = cloth_with_stained[start_x-h_stain//2:start_x+h_stain-h_stain//2, start_y-w_stain//2:start_y+w_stain-w_stain//2]    
                        
            #linear blend operator with coef alpha 
            alpha = np.random.uniform(0.3,1)
            output = (img_stain * alpha) + (roi * (1-alpha))
            output[mask3ch_stain == 0.] = roi[mask3ch_stain == 0.] 
            
            #overlay roi on cloth_image    
            cloth_with_stained[start_x-h_stain//2:start_x+h_stain-h_stain//2, start_y-w_stain//2:start_y+w_stain-w_stain//2] = output 
            mask_all_stain[start_x-h_stain//2:start_x+h_stain-h_stain//2, start_y-w_stain//2:start_y+w_stain-w_stain//2][mask3ch_stain==255.] = 255.
            
            #erase stain overlayed misleadingly on background
            cloth_with_stained[mask_cloth==0.] = img_cloth[mask_cloth==0.]
            mask_all_stain[mask_cloth==0.] = 0.

            break
        except : 
            count += 1
            if count >= 10:
                raise Exception("error randomize at border more than 10 times?")
    
    return cloth_with_stained, mask_all_stain

def generate_syn_stain_on_img(img_cloth, mask_cloth, mask_all_stain, hyp, algo_gen):
    
    cloth_with_stained = np.copy(img_cloth)
    h_img_cloth, w_img_cloth = cloth_with_stained.shape[:2]
    
    list_color = []
    list_type = []
    
    num_stain_circle = np.random.randint(low=hyp.get('min_number_stain'), high=hyp.get('max_number_stain'))
    for i in range(num_stain_circle):
        
        #random color for each stain inserted, make model more robust?
        color = generate_random_RGBcolor()
        list_color.append(color)

        #take algo gen from input
        algo_gen_copy = algo_gen
        if algo_gen_copy in ["random"]:
            algo_gen_copy = random.choice(["CircleV1", "CircleV2", "EdgeRad"])
        if algo_gen_copy in  ["CircleV1", "CircleV2"]:
            #generate coordinates of stain shape knowing n as number of control points
            n = np.random.randint(hyp.get('point_discretise_min'), hyp.get('point_discretise_max'))
            num_spike = n//np.random.randint(hyp.get("num_spike_coef1"),hyp.get("num_spike_coef2")) + np.random.randint(hyp.get("num_spike_coef3"), hyp.get("num_spike_coef4"))
            xi, yi =generate_deform_circle(n, num_spike, hyp)
        elif algo_gen_copy in ["EdgeRad"]:
            #generate coordinates of stain shape knowing n as number of control points
            n = np.random.randint(hyp.get('point_discretise_min'), hyp.get('point_discretise_max'))
            xi, yi =generate_random_shape(n)
        elif algo_gen_copy in ["OpenSimplexNoise"]:
            pass 
        else :
            raise Exception("Algo generate stain current {} not found", algo_gen_copy)            
        
        if not algo_gen_copy in ["OpenSimplexNoise"]:
            #make random axis distortion on shape
            xi, yi = distort_random_axis_shape(xi,yi)
            
            #translate coordinates to quadrant I (x,y positive) 
            if min(xi) < 0: 
                xi += abs(min(xi))
            if min(yi) < 0:
                yi += abs(min(yi))
    
            #scale the coordinates of stain to certain ratio with respect to cloth
            ratio_stain_on_cloth = np.random.uniform(low=hyp.get('ratio_stain_on_cloth_min'), high=hyp.get('ratio_stain_on_cloth_max')) 
            scale = min(h_img_cloth, w_img_cloth)//max(max(xi)-min(xi), max(yi)-min(yi))*ratio_stain_on_cloth
            xi *= scale
            yi *= scale
            
            #create image stain filled with color
            w, h = get_close_bbox_of_coords(xi, yi)[-2:]
            img_stain = np.zeros((h, w, 3))
            cv2.fillPoly(img_stain, pts=[np.array(np.vstack((xi,yi)).T, dtype=int)], color=color)
        
        else : #algo opensimplexnoise
            ratio_stain_on_cloth = np.random.uniform(low=hyp.get('ratio_stain_on_cloth_min'), high=hyp.get('ratio_stain_on_cloth_max')) 
            img_stain_size = int(min(h_img_cloth, w_img_cloth)*ratio_stain_on_cloth*3) #3 as the default ratio_box_stain of function generate_circle_opensimplexnoise
            noise_level = np.random.uniform(low=hyp.get('noise_level_min'), high=hyp.get('noise_level_max')) 
            mask_stain = generate_circle_opensimplexnoise(img_stain_size, noise_level=noise_level)
            img_stain = np.zeros((img_stain_size, img_stain_size, 3))
            img_stain[mask_stain==255] = color
            
        #get mask of stain 
        img_stain = img_stain.astype(np.uint8)
        stain_gray = cv2.cvtColor(img_stain, cv2.COLOR_RGB2GRAY)
        invert_3ch = get_mask3channels_stainraw(stain_gray)
        invert_3ch = cv2.bitwise_not(invert_3ch)
        
        #get new image by insert stain on, with additional binary mask indicating position of stain on cloth
        cloth_with_stained, mask_all_stain = insert_stain_raw_in_img(cloth_with_stained, img_stain, invert_3ch, mask_cloth, mask_all_stain)
        
        #add type of stains to list
        list_type.append(algo_gen_copy)
        
    return cloth_with_stained, mask_all_stain, list_color, list_type
