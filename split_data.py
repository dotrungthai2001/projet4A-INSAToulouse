

import os, shutil, random, glob
random.seed(0) 

# preparing the folder structure

full_data_path = 'data/DATA/'
extension_allowed = '.jpg'
split_percentage = 90

images_path = 'data/images/'
if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path)
    
labels_path = 'data/labels/'
if os.path.exists(labels_path):
    shutil.rmtree(labels_path)
os.mkdir(labels_path)
    
training_images_path = images_path + 'training/'
validation_images_path = images_path + 'validation/'
training_labels_path = labels_path + 'training/'
validation_labels_path = labels_path +'validation/'
    
os.mkdir(training_images_path)
os.mkdir(validation_images_path)
os.mkdir(training_labels_path)
os.mkdir(validation_labels_path)

all_images = []
ext_len = len(extension_allowed)
for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            all_images.append(file)

# get a list of all the labeled image files in the data directory
labeled_image_files = []
for f in os.listdir(full_data_path):
    f_extension = f.split(".")[-1]
    f_name_no_extension = f.replace("."+f_extension, "")
    if f.endswith(extension_allowed) and os.path.isfile(os.path.join(full_data_path, f_name_no_extension + ".txt")):
        labeled_image_files.append(f)

# get a list of all the background image files in the data directory
background_image_files = []
for f in os.listdir(full_data_path):
    f_extension = f.split(".")[-1]
    f_name_no_extension = f.replace("."+f_extension, "")
    if f.endswith(extension_allowed) and not os.path.isfile(os.path.join(full_data_path, f_name_no_extension + ".txt")):
        background_image_files.append(f)

assert sorted(all_images) == sorted(labeled_image_files + background_image_files)


# shuffle the labeled image files and background image files separately
random.shuffle(labeled_image_files)
random.shuffle(background_image_files)

# calculate the number of labeled and background images to include in the training set
num_labeled_training_images = int(len(labeled_image_files) * split_percentage / 100)
num_background_training_images = int(len(background_image_files) * split_percentage / 100)

# split the labeled image files and background image files into training and validation sets
labeled_training_images = sorted(labeled_image_files[:num_labeled_training_images])
labeled_validation_images = sorted(labeled_image_files[num_labeled_training_images:])
background_training_images = sorted(background_image_files[:num_background_training_images])
background_validation_images = sorted(background_image_files[num_background_training_images:])

print("***copying training data")
print("copy the labeled training images to the training file : ", len(labeled_training_images))

for image_file in labeled_training_images:
    image_name_without_format = image_file.split(".")[0]
    
    #copy the image file
    image_path = full_data_path + image_file
    shutil.copy(image_path, training_images_path) 
    
    #copy the label file
    annotation_file = image_name_without_format + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, training_labels_path) 
    
print("copy the background training images to the training file : ", len(background_training_images))
for image_file in background_training_images:
    image_path = full_data_path + image_file
    # copy the image file
    shutil.copy(image_path, training_images_path) 

print("***copying validation data")
print("copy the labeled val images to the val file : ", len(labeled_validation_images))

for image_file in labeled_validation_images:
    image_name_without_format = image_file.split(".")[0]
    
    #copy the image file
    image_path = full_data_path + image_file
    shutil.copy(image_path, validation_images_path) 
    
    #copy the label file
    annotation_file = image_name_without_format + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, validation_labels_path) 

print("copy the background training images to the training file : ", len(background_validation_images))
for image_file in background_validation_images:
    image_path = full_data_path + image_file
    # copy the image file
    shutil.copy(image_path, validation_images_path) 
    
print("finished")
