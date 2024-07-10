# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:51:16 2023
Refactored on Thu Mar 23 10:10:50 2023
@author: TranK
"""

import argparse
from utils import *
import json
from augmentation_background import *
import pickle

def main(args):

    #load arguments
    imgs_folder = args.clothes_folder
    backs_folder = args.backs_folder

    #get paste_to directory (directory store new data created)
    paste_to = args.paste_to_folder
    paste_to = os.path.join(paste_to, "DATA_syn_stain/")
    paste_to_mask = os.path.join(paste_to, "MASK_syn_stain/")
    if not os.path.exists(paste_to):
        os.mkdir(paste_to)
    if not os.path.exists(paste_to_mask):
        os.mkdir(paste_to_mask)
    print("****Folder where new data are created : ", paste_to)

    # get imgs/backs paths list
    imgs_paths  = list_full_paths(imgs_folder)
    if backs_folder:
        backs_paths = list_full_paths(backs_folder)

    # if use of subset 
    subset_use = args.subset_use
    if subset_use:
        imgs_paths = imgs_paths[:subset_use]
        print("****Attention, only {} images are used when calling subset_use".format(subset_use))
    else:
        print("****All images from {} will be used to create new dataset".format(imgs_folder))

    #identify images with object to insert and images without
    random.shuffle(imgs_paths)
    ratio_no_stain = 0.2 #ratio will be divided by number of types of algo taken
    num_cloth_insert = int(len(imgs_paths) * (1-ratio_no_stain))
    imgs_insert_paths = sorted(imgs_paths[:num_cloth_insert])
    imgs_no_insert_paths = sorted(imgs_paths[num_cloth_insert:])

    #copy images without objects to dataset
    for no, image_path in enumerate(imgs_no_insert_paths):
        name_img = image_path.split("/")[-1]
        shutil.copy(image_path, paste_to + name_img)
    print("****Number of images with stains inserted = ", len(imgs_insert_paths))
    print("****Number of images without objects (stains) = ", len(imgs_no_insert_paths))

    #algo generator (3 types)
    algo_gen_all = ["CircleV1", "CircleV2", "EdgeRad", "OpenSimplexNoise"]
    hyp_algo_yaml = ["hyp.CircleV1.yaml", 
                     "hyp.CircleV2.yaml", 
                     "hyp.EdgeRad.yaml", 
                     "hyp.OpenSimplexNoise.yaml"]
    
    #print(args.algo_gen)
    algo_gen_list = args.algo_gen
    print("****Algo gen runned", algo_gen_list)

    #dict to store stats of color and area of stains created
    dict_color = dict()
    dict_area = dict()
    dict_position = dict()
    dict_type = dict()

    iter_current = 0
    for algo_gen in algo_gen_list:
        #hyperparameters
        hyp = hyp_algo_yaml[algo_gen_all.index(algo_gen)]
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        
        #generate stains on images
        for no, image_path in enumerate(imgs_insert_paths):   
            iter_current += 1
            if iter_current%50 == 0 :
                print("**Currently insert image number : ", iter_current)
            img_cloth = cv2.imread(image_path)
            
            #return mask of cloth (foreground)
            mask_cloth_rembg = remove(img_cloth, only_mask=True, post_process_mask=True) #binary mask [0,255]
            
            #generate new images with stains inserted and a binary mask for stains inserted position
            mask_all_stain = np.zeros_like(img_cloth)
            cloth_with_stained, mask_all_stain, list_color, list_type = generate_syn_stain_on_img(img_cloth, mask_cloth_rembg, mask_all_stain, hyp, algo_gen)
            dict_color[algo_gen+"_"+name_img] = list_color
            dict_type[algo_gen+"_"+name_img] = list_type
            #augmented background
            if backs_folder :
                if np.random.uniform(0,1) < 0.3: 
                    img_background = cv2.imread(np.random.choice(backs_paths))
                    cloth_with_stained_without_bg = get_cloth_without_bg(cloth_with_stained, mask_cloth_rembg)
                    cloth_with_stained = add_shadow_and_background(cloth_with_stained_without_bg, mask_cloth_rembg, img_background)
 
            #write images with stains added
            name_img = image_path.split("/")[-1]
            cv2.imwrite(paste_to + algo_gen + "_"+ name_img, cloth_with_stained)

            # #write mask of stains for each image
            # cv2.imwrite(paste_to_mask + algo_gen + "_"+ name_img, mask_all_stain)
        
            #find bounding boxes based on binary mask : stains overlapsed belong to same box
            lbl_0 = label(mask_all_stain[:, :, 0]) 
            props = regionprops(lbl_0)
            stain_box_all_skimage = [prop.bbox for prop in props]
            stain_box_all = convert_SkimageToYoloFormat(stain_box_all_skimage, img_cloth.shape[0], img_cloth.shape[1])
            
            list_area = []
            list_position = []
            for box in stain_box_all_skimage:
                ratio_area = compute_ratio_number_pixels(mask_all_stain[box[0]:box[2],box[1]:box[3]], img_cloth.shape[0], img_cloth.shape[1])
                list_area.append(ratio_area)
                stain_pos = [(box[0]+box[2])//2 , (box[1]+box[3])//2]
                list_position.append(stain_pos)
            dict_area[algo_gen+"_"+name_img] = list_area
            dict_position[algo_gen+"_"+name_img] = list_position
            
            #create annotations files corresponding
            name_img_noFileFormat = name_img.split(".")[0]
            with open(paste_to + algo_gen + "_" + name_img_noFileFormat+ ".txt",'w+') as f:
                for stain_box in stain_box_all:
                    assert len(stain_box) == 4
                    f.write("0 %f " % stain_box[0])
                    f.write("%f " % stain_box[1])
                    f.write("%f " % stain_box[2])
                    f.write("%f\n" % stain_box[3])  

    #store stats of color and area of stains in pickle files  
    pickle_folder = os.path.join(paste_to, "pickle_folder/")
    if not os.path.exists(pickle_folder):
        os.mkdir(pickle_folder)
    with open(pickle_folder+"dict_color"+".pkl","wb") as g:
        pickle.dump(dict_color, g)
    with open(pickle_folder+"dict_area"+".pkl","wb") as h:
        pickle.dump(dict_area, h) 
    with open(pickle_folder+"dict_position"+".pkl","wb") as k:
        pickle.dump(dict_area, k)
    with open(pickle_folder+"dict_type"+".pkl","wb") as m:
        pickle.dump(dict_area, m)
if __name__ == '__main__':
    arguments = parse()
    main(arguments)
