# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:51:16 2023
@author: TranK
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
from scipy.interpolate import interp1d
from rembg import remove
from scipy.special import binom
from helper_functions import *
from parameters import *
from add_shadow import * 
import random, shutil
from skimage.measure import label, regionprops

def random_spaced(low, high, delta, n, size=None):
    #https://stackoverflow.com/questions/53565205/how-to-generate-random-numbers-with-each-random-number-having-a-difference-of-at
    empty_space = high - low - (n-1)*delta
    if empty_space < 0:
        raise ValueError("not possible")
    if size is None:
        u = np.random.rand(n)
    else:
        u = np.random.rand(size, n)
    x = empty_space * np.sort(u, axis=-1)
    return np.ceil(low + x + delta * np.arange(n)).astype(int)

def bounding_box(x_list, y_list):
    return np.ceil(np.array([min(x_list), min(y_list), max(x_list)-min(x_list), max(y_list)-min(y_list)])).astype(int)

def convert_boundingbox_format (bbox, h_image, w_image):
    h_image, w_image = float(h_image), float(w_image)
    x_center, y_center, w_box, h_box = [float(bbox[i]) for i in range(4)]  
    return [x_center/w_image, y_center/h_image, w_box/w_image, h_box/h_image]

def convert_SkimageToYoloFormat(list_bb_skimage, h_image, w_image):
    h_image, w_image = float(h_image), float(w_image)

    list_bb_yolo = [] 
    for box in list_bb_skimage:
        y_min, x_min, y_max, x_max = [i for i in box]
        w_box, h_box = x_max - x_min, y_max - y_min
        x_center, y_center = (x_min+w_box/2., y_min+h_box/2.)
        list_bb_yolo.append([x_center/w_image, y_center/h_image, w_box/w_image, h_box/h_image])
    return list_bb_yolo

def get_mask3channels_stainraw(stain_gray):
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

def rotate_random_axis(xi, yi,min_distort_coef=1.,max_distort_coef=4.):
    distort_coef = np.random.uniform(min_distort_coef,max_distort_coef)
    distort_angle = np.random.uniform()*2*np.pi
    
    yi /= distort_coef
    new_xi = xi*np.cos(distort_angle) + yi*np.sin(distort_angle)
    new_yi = -xi*np.sin(distort_angle) + yi*np.cos(distort_angle)
    return new_xi, new_yi

def compute_ratio_number_pixels(mask_stain, mask_cloth):
    gray_stain = cv2.cvtColor(mask_stain, cv2.COLOR_RGB2GRAY)
    gray_cloth = cv2.cvtColor(mask_cloth, cv2.COLOR_RGB2GRAY)
    return float(cv2.countNonZero(gray_stain))/float(cv2.countNonZero(gray_cloth))

def insert_stain_raw_in_img(img_cloth, img_stain, mask3ch_stain, mask_cloth, mask_all_stain):
    assert img_cloth.shape == mask_all_stain.shape
    cloth_with_stained = np.copy(img_cloth)
    h_cloth, w_cloth = cloth_with_stained.shape[:2]
    
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
            
            #add average color of clothes to stain
            #color_bg = np.mean(roi, axis=(0,1))  
            #img_bg_average = np.full((h_stain, w_stain, 3), color_bg, dtype=('uint8'))
            #img_new = cv2.addWeighted(img_stain,0.8,img_bg_average,0.2,0)
            
            #alpha blending 
            alpha = np.random.uniform(0.25,1)
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

def Deform_circle_Ver2(n=50, 
                  num_spike=6, 
                  periodic=False,
                  noise_spike=False,
                  noise_circle=False):
        
    theta = 2 * np.pi * np.linspace(0, 1, n)
    max_spike_coef = 1.6
    min_spike_coef = 1.2
    max_noise = 1.08
    min_noise = 0.92
    
    if periodic == True: #periodic creation of spike around the circle 
        period = 2*n//(num_spike)

        index_spike = np.array([i*(period-1) for i in range((n-2)//(period-1)+1)])
        index_neighbor = index_spike + 1
        
        index_spike = np.concatenate([index_spike, index_neighbor])
        index_spike = np.unique(index_spike)

        not_spike = np.array([i for i in range(n-2) if i not in index_spike]) 
                
    else: #not periodic, generate random spike
        
        index_spike = random_spaced(low=1, high=n-3, delta=3, n=num_spike//2, size=None)
        index_neighbor = index_spike + 1
        
        index_spike = np.concatenate([index_spike, index_neighbor])
        index_spike = np.unique(index_spike)
        
        not_spike = np.array([i for i in range(n-2) if i not in index_spike])
    
    random_noise = np.random.uniform(low=min_spike_coef, high=max_spike_coef, size=(n-2,2))
    random_noise[not_spike] =  np.random.uniform(low=min_noise, high=max_noise, size=(len(not_spike), 2))
        
    y = np.c_[np.cos(theta), np.sin(theta)]
    y[1:-1, :] = np.c_[np.cos(theta), np.sin(theta)][1:-1, :] * random_noise
    
    cs = CubicSpline(theta, y, bc_type='periodic')
    xs = 2 * np.pi * np.linspace(0, 1, n)
 
    # if noise_spike == True: #add noise to spike 
    #     y_noise = np.copy(y)
        
    #     ###add noise to 2 points root of spike
    #     index_add_noise = np.concatenate([index_spike - 1, index_spike +1]) 
    #     index_add_noise = np.sort(index_add_noise)
        
    #     ###no noise if spike too close
    #     def drop_close_value_array(list_index, tolerance=3):
    #         new_index = []
    #         start = index_add_noise[0]
    #         for index in index_add_noise[1:]:
    #             if abs (start-index) >= tolerance:
    #                 new_index.append(index)
    #                 start = index
    #         return np.array(new_index)
        
    #     index_add_noise = drop_close_value_array(index_add_noise)
        
    #     ###no noise at first&last element, first element must equal last element for a close shape
    #     index_add_noise = np.delete(index_add_noise, np.where(index_add_noise<0))
    #     index_add_noise = np.delete(index_add_noise, np.where(index_add_noise>=n-2))

    #     y_noise[1+index_add_noise, 0] *=  np.random.uniform(0.8, 1.5, size=(len(index_add_noise)))
    #     y_noise[1+index_add_noise, 1] *=  np.random.uniform(0.8, 1.5, size=(len(index_add_noise)))
        
    #     cs = CubicSpline(theta, y_noise, bc_type='periodic')
    #     xs = 2 * np.pi * np.linspace(0, 1, n)
    
    # else : #no noise to spike, the spike have equal width, different height
    #     cs = CubicSpline(theta, y, bc_type='periodic')
    #     xs = 2 * np.pi * np.linspace(0, 1, n)
        
    # if noise_circle:
    #     points_delete = n//5
    #     index_point_delete = np.random.randint(low = 1, high = n-2, size = points_delete)
        
    #     theta = np.delete(theta, index_point_delete)
    #     y = np.delete(y, index_point_delete, axis=0)
        
    #     cs =CubicSpline(theta, y, bc_type='periodic')
    #     xs = 2 * np.pi * np.linspace(0, 1, n)
    #     xs = np.delete(xs, index_point_delete)

    
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(y[:, 0], y[:, 1], 'o')
    # ax.plot(np.cos(xs), np.sin(xs), '*', label='true')
    # ax.plot(cs(xs)[:, 0], cs(xs)[:, 1])
    # # ax.plot(y_noise[:, 0], y_noise[:, 1], 'o')
    # ax.axes.set_aspect('equal')
    # ax.legend(loc='center')
    # plt.show()
    
    x = cs(xs)[:, 0]
    y = cs(xs)[:, 1]
    
    t = np.arange(len(x))
    ti = np.linspace(0, t.max(), 10 * t.size)
    
    xi = interp1d(t, x, kind='cubic')(ti)
    yi = interp1d(t, y, kind='cubic')(ti)
    
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(xi, yi)
    # # ax.plot(x, y)
    # # ax.margins(0.05)
    # plt.show()
        
    return xi, yi

def Deform_circle_Ver3(n=50, 
                  num_spike=6, 
                  periodic=False,
                  noise_spike=False,
                  noise_circle=False):
        
    theta = 2 * np.pi * np.linspace(0, 1, n)
    max_spike_coef = 1.6
    min_spike_coef = 1.1
    max_noise = 1.02
    min_noise = 0.97
    
    if periodic == True: #periodic creation of spike around the circle 
        period = 2*n//(num_spike)

        index_spike = np.array([i*(period-1) for i in range((n-2)//(period-1)+1)])
        index_neighbor = index_spike + 1
        
        index_spike = np.concatenate([index_spike, index_neighbor])
        index_spike = np.unique(index_spike)

        not_spike = np.array([i for i in range(n-2) if i not in index_spike]) 
                
    else: #not periodic, generate random spike
        index_spike = random_spaced(low=1, high=n-3, delta=3, n=num_spike//2, size=None)
        index_neighbor = index_spike + 1
        
        index_spike = np.concatenate([index_spike, index_neighbor])
        index_spike = np.unique(index_spike)
        
        not_spike = np.array([i for i in range(n-2) if i not in index_spike])
    
    random_noise = np.random.uniform(low=min_spike_coef, high=max_spike_coef, size=(n-2,2))
    random_noise[not_spike] =  np.random.uniform(low=min_noise, high=max_noise, size=(len(not_spike), 2))
        
    y = np.c_[np.cos(theta), np.sin(theta)]
    y[1:-1, :] = np.c_[np.cos(theta), np.sin(theta)][1:-1, :] * random_noise
    
    cs = CubicSpline(theta, y, bc_type='periodic')
    xs = 2 * np.pi * np.linspace(0, 1, n)
    
    x = cs(xs)[:, 0]
    y = cs(xs)[:, 1]
    
    t = np.arange(len(x))
    ti = np.linspace(0, t.max(), 10 * t.size)
    
    xi = interp1d(t, x, kind='cubic')(ti)
    yi = interp1d(t, y, kind='cubic')(ti)
        
    return xi, yi

def insert_circleStain_on_image_Ver2(img_cloth, mask_cloth, mask_all_stain, num_stain_circle, periodic=False, noise_spike=False, noise_circle=False):
    
    # color = color_list[np.random.randint(0, len(color_list))]
    cloth_with_stained = np.copy(img_cloth)
    h_img_cloth, w_img_cloth = cloth_with_stained.shape[:2]
    
    point_discretise_min = 35
    point_discretise_max = 45
    
    list_color = []
    
    for i in range(num_stain_circle):
        
        #random color for each stain inserted, make model more robust?
        r = np.random.randint(0,255)
        g = np.random.randint(0,255)
        b = np.random.randint(0,255)
        color = [r,g,b]
        if color not in list_color:
            list_color.append(color)
        n = np.random.randint(point_discretise_min, point_discretise_max)
        num_spike = n//np.random.randint(4,6) + np.random.randint(3, 7)
        xi, yi =Deform_circle_Ver2(n, num_spike, periodic=periodic, noise_spike=noise_spike,  noise_circle=noise_circle)
        xi, yi = rotate_random_axis(xi,yi)
        
        if min(xi) < 0: 
            xi += abs(min(xi))
        if min(yi) < 0:
            yi += abs(min(yi))
            
        scale = min(h_img_cloth, w_img_cloth)//max(max(xi)-min(xi), max(yi)-min(yi))//np.random.randint(5, 12)
        xi *= scale
        yi *= scale
        
        start_x, start_y, w, h = bounding_box(xi, yi)
        img_stain = np.zeros((h, w, 3))
        cv2.fillPoly(img_stain, pts=[np.array(np.vstack((xi,yi)).T, dtype=int)], color=color)
        
        img_stain = img_stain.astype(np.uint8)
        stain_gray = cv2.cvtColor(img_stain, cv2.COLOR_RGB2GRAY)
        invert_3ch = get_mask3channels_stainraw(stain_gray)
        invert_3ch = cv2.bitwise_not(invert_3ch)
        
        cloth_with_stained, mask_all_stain = insert_stain_raw_in_img(cloth_with_stained, img_stain, invert_3ch, mask_cloth, mask_all_stain)
        
    return cloth_with_stained, mask_all_stain, list_color

def insert_RandomShape_on_image(img_cloth, mask_cloth, mask_all_stain, num_stain_circle):
    # color = color_list[np.random.randint(0, len(color_list))]
    cloth_with_stained = np.copy(img_cloth)
    h_img_cloth, w_img_cloth = cloth_with_stained.shape[:2]
    
    point_discretise_min = 10
    point_discretise_max = 40
    
    list_color = []

    for i in range(num_stain_circle):
        
        #random color for each stain inserted, make model more robust?
        r = np.random.randint(0,255)
        g = np.random.randint(0,255)
        b = np.random.randint(0,255)
        color = [r,g,b]
        if color not in list_color:
            list_color.append(color)
        
        n = np.random.randint(point_discretise_min,point_discretise_max)
        a = get_random_points(n, scale=9)
        
        rad = np.random.choice([0.2, 0.3, 0.4])
        edgy = np.random.choice([0., 0.01])
        xi, yi = get_bezier_curve(a, rad=rad, edgy=edgy)[:2]
        xi, yi = rotate_random_axis(xi,yi)
        
        if min(xi) < 0: 
            xi += abs(min(xi))
        if min(yi) < 0:
            yi += abs(min(yi))
            
        scale = min(h_img_cloth, w_img_cloth)//max(max(xi)-min(xi), max(yi)-min(yi))//np.random.randint(5, 12)
        xi *= scale
        yi *= scale
        start_x, start_y, w, h = bounding_box(xi, yi)
        
        img_stain = np.zeros((h, w, 3))
        cv2.fillPoly(img_stain, pts=[np.array(np.vstack((xi,yi)).T, dtype=int)], color=color)
        
        img_stain = img_stain.astype(np.uint8)
        stain_gray = cv2.cvtColor(img_stain, cv2.COLOR_RGB2GRAY)
        invert_3ch = get_mask3channels_stainraw(stain_gray)
        invert_3ch = cv2.bitwise_not(invert_3ch)
    
        cloth_with_stained, mask_all_stain = insert_stain_raw_in_img(cloth_with_stained, img_stain, invert_3ch, mask_cloth, mask_all_stain)
        
    return cloth_with_stained, mask_all_stain, list_color

def insert_circleStain_on_image_Ver3(img_cloth, mask_cloth, mask_all_stain, num_stain_circle, 
                                     color_list=[(153,51,0), (0,102,204), (51,51,51),(55, 78, 111)],
                                     periodic=False, noise_spike=False, noise_circle=True):
    
    cloth_with_stained = np.copy(img_cloth)
    h_img_cloth, w_img_cloth = cloth_with_stained.shape[:2]
    
    point_discretise_min = 80
    point_discretise_max = 180
    
    list_color = []
    
    for i in range(num_stain_circle):
        
        #random color for each stain inserted, make model more robust?
        r = np.random.randint(0,255)
        g = np.random.randint(0,255)
        b = np.random.randint(0,255)
        color = [r,g,b]
        if color not in list_color:
            list_color.append(color)

        n = np.random.randint(point_discretise_min, point_discretise_max)
        num_spike = n//np.random.randint(4,6) +  np.random.randint(5, 20)
        xi, yi =Deform_circle_Ver3(n, num_spike, periodic=periodic, noise_spike=noise_spike,  noise_circle=noise_circle) 
        xi, yi = rotate_random_axis(xi,yi)
        
        if min(xi) < 0: 
            xi += abs(min(xi))
        if min(yi) < 0:
            yi += abs(min(yi))
            
        scale = min(h_img_cloth, w_img_cloth)//max(max(xi)-min(xi), max(yi)-min(yi))//np.random.randint(5, 12)
        xi *= scale
        yi *= scale
        start_x, start_y, w, h = bounding_box(xi, yi)

        img_stain = np.zeros((h, w, 3))
        cv2.fillPoly(img_stain, pts=[np.array(np.vstack((xi,yi)).T, dtype=int)], color=color)
        
        img_stain = img_stain.astype(np.uint8)
        stain_gray = cv2.cvtColor(img_stain, cv2.COLOR_RGB2GRAY)
        invert_3ch = get_mask3channels_stainraw(stain_gray)
        invert_3ch = cv2.bitwise_not(invert_3ch)
    
        cloth_with_stained, mask_all_stain = insert_stain_raw_in_img(cloth_with_stained, img_stain, invert_3ch, mask_cloth, mask_all_stain)
        
    return cloth_with_stained, mask_all_stain, list_color

######################################################
path_test = path_base + cloth_folder
path_background = path_base + background_folder

paste_to = path_base +"DATA/"
if not os.path.exists(paste_to):
    os.mkdir(paste_to)
        
annotation = path_base +"MASK/"
if not os.path.exists(annotation):
    os.mkdir(annotation)

images_test = glob.glob(path_test + "*")
background_images = glob.glob(path_background +"*")

#window glob adjust path \\ to /
images = [image_path.replace("\\", "/") for image_path in images_test]
background_path_set = [image_path.replace("\\", "/") for image_path in background_images]

COMPARISON_NUM = 500
images = images[:COMPARISON_NUM]

#split images into 90% to insert and 10% cloth only
random.shuffle(images)
split_percentage = 90
num_cloth_insert = len(images) - int(len(images) * (1-split_percentage / 100)*3/2)
images_insert = sorted(images[:num_cloth_insert])
images_background_only = sorted(images[num_cloth_insert:])

#copy background only image to dataset
for no, image_path in enumerate(images_background_only):
     print("Image no ", no)
     name_img = image_path.split("/")[-1]
     shutil.copy(image_path, paste_to + name_img)

#generate dataset with different algo
deform_circle_add_V2 = False
deform_circle_add_V3 = False
random_shape_edgy_add = True

dict_color = dict()
dict_area = dict()
######################################################
if deform_circle_add_V2:        
    for no, image_path in enumerate(images_insert):   
        print("Image no ", no)
        img_cloth = cv2.imread(image_path)
        img_background = cv2.imread(np.random.choice(background_path_set))
        mask_cloth_rembg = remove(img_cloth, only_mask=True, post_process_mask=True) #binary mask [0,255]
        
        mask_all_stain = np.zeros_like(img_cloth)
        num_stain_circle = np.random.randint(2, 4)
        cloth_with_stained, mask_all_stain, list_color = insert_circleStain_on_image_Ver2(img_cloth, mask_cloth_rembg, mask_all_stain,num_stain_circle=num_stain_circle, periodic=False, noise_spike=False, noise_circle=False)
        
        if np.random.uniform(0,1) < 0.3: 
            cloth_with_stained_without_bg = get_cloth_without_bg(cloth_with_stained, mask_cloth_rembg)
            cloth_with_stained = add_shadow_and_background(cloth_with_stained_without_bg, mask_cloth_rembg, img_background)
        
        name_img = image_path.split("/")[-1]
        
        dict_color['V2_circle_deform_'+name_img] = list_color
        #write images with stains added
        cv2.imwrite(paste_to + "V2_circle_deform_"+ name_img, cloth_with_stained)
        
        ##write mask of stains on clothes
        # cv2.imwrite(annotation + "V2_circle_deform_"+ name_img, mask_all_stain)
        
        #create text file for bounding boxes with class 0 : stain
        lbl_0 = label(mask_all_stain[:, :, 0]) 
        #plt.imshow(lbl_0)
        props = regionprops(lbl_0)
        stain_box_all_skimage = [prop.bbox for prop in props]
        
        list_area = []
        for box in stain_box_all_skimage:
            ratio_area = compute_ratio_number_pixels(mask_all_stain[box[0]:box[2],box[1]:box[3]], mask_cloth_rembg)
            list_area.append(ratio_area)
        dict_area['V2_circle_deform_'+name_img] = list_area
        
        # for box in stain_box_all_skimage:
        #     cv2.rectangle(cloth_with_stained, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 5)
        stain_box_all = convert_SkimageToYoloFormat(stain_box_all_skimage, img_cloth.shape[0], img_cloth.shape[1])
        
        name_img_noFileFormat = name_img.split(".")[0]
        with open(paste_to + "V2_circle_deform_"+ name_img_noFileFormat+ ".txt",'w+') as f:
            for stain_box in stain_box_all:
                assert len(stain_box) == 4
                f.write("0 %f " % stain_box[0])
                f.write("%f " % stain_box[1])
                f.write("%f " % stain_box[2])
                f.write("%f\n" % stain_box[3])
        

######################################################
if deform_circle_add_V3:        
    for no, image_path in enumerate(images_insert):   
        print("Image no ", no)
        img_cloth = cv2.imread(image_path)
        img_background = cv2.imread(np.random.choice(background_path_set))
        
        mask_cloth_rembg = remove(img_cloth, only_mask=True, post_process_mask=True) #binary mask [0,255]
        
        mask_all_stain = np.zeros_like(img_cloth)
        num_stain_circle = np.random.randint(1, 7)
        cloth_with_stained, mask_all_stain, list_color = insert_circleStain_on_image_Ver3(img_cloth, mask_cloth_rembg, mask_all_stain, num_stain_circle=num_stain_circle, periodic=False, noise_spike=False, noise_circle=False)
        
        if np.random.uniform(0,1) < 0.3: 
            cloth_with_stained_without_bg = get_cloth_without_bg(cloth_with_stained, mask_cloth_rembg)
            cloth_with_stained = add_shadow_and_background(cloth_with_stained_without_bg, mask_cloth_rembg, img_background)
        
        name_img = image_path.split("/")[-1]
        
        dict_color['V3_circle_deform_'+name_img] = list_color
        
        cv2.imwrite(paste_to + "V3_circle_deform_"+ name_img, cloth_with_stained)
        #cv2.imwrite(annotation + "V3_circle_deform_"+ name_img, mask_all_stain)
        
        #create text file for bounding boxes with class 0 : stain
        lbl_0 = label(mask_all_stain[:, :, 0]) 
        props = regionprops(lbl_0)
        stain_box_all_skimage = [prop.bbox for prop in props]
        # for box in stain_box_all_skimage:
        #     cv2.rectangle(cloth_with_stained, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 5)

        stain_box_all = convert_SkimageToYoloFormat(stain_box_all_skimage, img_cloth.shape[0], img_cloth.shape[1])
        
        list_area = []
        for box in stain_box_all_skimage:
            ratio_area = compute_ratio_number_pixels(mask_all_stain[box[0]:box[2],box[1]:box[3]], mask_cloth_rembg)
            list_area.append(ratio_area)
        dict_area['V3_circle_deform_'+name_img] = list_area

        name_img_noFileFormat = name_img.split(".")[0]
        with open(paste_to + "V3_circle_deform_"+ name_img_noFileFormat+ ".txt",'w+') as f:
            for stain_box in stain_box_all:
                assert len(stain_box) == 4
                f.write("0 %f " % stain_box[0])
                f.write("%f " % stain_box[1])
                f.write("%f " % stain_box[2])
                f.write("%f\n" % stain_box[3])
                
######################################################
if random_shape_edgy_add:        
    for no, image_path in enumerate(images_insert):   
        print("Image no ", no)
        img_cloth = cv2.imread(image_path)
        img_background = cv2.imread(np.random.choice(background_path_set))
        
        mask_cloth_rembg = remove(img_cloth, only_mask=True, post_process_mask=True) #binary mask [0,255]
        
        mask_all_stain = np.zeros_like(img_cloth)
        num_stain_circle = np.random.randint(1, 4)
        cloth_with_stained, mask_all_stain, list_color = insert_RandomShape_on_image(img_cloth, mask_cloth_rembg, mask_all_stain, num_stain_circle=num_stain_circle)
        
        if np.random.uniform(0,1) < 0.3: 
            cloth_with_stained_without_bg = get_cloth_without_bg(cloth_with_stained, mask_cloth_rembg)
            cloth_with_stained = add_shadow_and_background(cloth_with_stained_without_bg, mask_cloth_rembg, img_background)
          
        name_img = image_path.split("/")[-1]
        
        dict_color['random_shape_edgy'+name_img] = list_color

        cv2.imwrite(paste_to + "random_shape_edgy"+ name_img, cloth_with_stained)
        #cv2.imwrite(annotation + "random_shape_edgy"+ name_img, mask_all_stain)
        
        #create text file for bounding boxes with class 0 : stain
        lbl_0 = label(mask_all_stain[:, :, 0]) 
        props = regionprops(lbl_0)
        stain_box_all_skimage = [prop.bbox for prop in props]
        # for box in stain_box_all_skimage:
        #     cv2.rectangle(cloth_with_stained, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 5)

        stain_box_all = convert_SkimageToYoloFormat(stain_box_all_skimage, img_cloth.shape[0], img_cloth.shape[1])

        list_area = []
        for box in stain_box_all_skimage:
            ratio_area = compute_ratio_number_pixels(mask_all_stain[box[0]:box[2],box[1]:box[3]], mask_cloth_rembg)
            list_area.append(ratio_area)
        dict_area['random_shape_edgy'+name_img] = list_area
        
        name_img_noFileFormat = name_img.split(".")[0]
        with open(paste_to + "random_shape_edgy"+ name_img_noFileFormat+ ".txt",'w+') as f:
            for stain_box in stain_box_all:
                assert len(stain_box) == 4
                f.write("0 %f " % stain_box[0])
                f.write("%f " % stain_box[1])
                f.write("%f " % stain_box[2])
                f.write("%f\n" % stain_box[3])

###################################################### 
# Afer 3 for loops above :
    # color = [r,g,b]
with open(paste_to+"dict_color"+".pkl","wb") as g:
    pickle.dump(dict_color, g)
with open(paste_to+"dict_area"+".pkl","wb") as h:
    pickle.dump(dict_area, h) 
