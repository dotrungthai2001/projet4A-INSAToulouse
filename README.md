## projet4A

# Link to data, doc and code

Link to git https://github.com/tkdang-insa/projet4A

Link to drive for writing https://drive.google.com/drive/folders/1vkSDrD5i1twVNnZ7hs94Ky-1yHiGIekT

Link to dataset of tshirt  https://we.tl/t-pVZ9hGAMGj

Link to dataset of tshirt with stain inserted : https://drive.google.com/drive/folders/1v6ppYdjuz5pwaDALfdVDBAfbZ_1Sak31?usp=sharing

# Structure of working directory

Structure of data and code are organised as below:

<pre>
├───yolov5_ws
│   ├───data 
│   │   ├───data_cloth_only (1218 tshirst no stain)
│   │   ├───data_stain_only (7 real stains collected on web)
│   │   ├───DATA (clothes with stain inserted - 4872 images with 4872 txt file bbox annotations)
│   │   ├───MASK (4872 binary masks)
│   │   ├───images
│   │   │   ├───training (90%)
│   │   │   ├───validation (10%)
│   │   ├───label
│   │   │   ├───training (90%)
│   │   │   ├───validation (10%)
│   ├───projet4A (code for synthetic data creation)
│   │   ├───parameters.py
│   │   ├───helper_functions.py
│   │   ├───synthethic_data_ver_khoi.py
│   ├───dataset.yaml
│   ├───split_data.py
│   ├───yolov5 (code for model training)
│   │   ├───requirements.txt
│   │   ├───train.py
│   │   ├───val.py
│   │   ├───detect.py
│   │   ├───...
└───__pycache__
</pre>

# PART 1 : Data synthetic create 

main file to run : synthetic_data_ver_khoi.py
file set up working directory : parameters.py

Make sure that directory on your machine are filled in parameters.py (don't push it on git once changed)
e.g. make sure you have the folder of cloth and the folder of stain. Make sure you have folder data_cloth englobe in a folder dataset (c.f. structure of data code).

Next, we run the file synthetic_data_ver_khoi.py, it will create 2 new folderes DATA (image cloth with stain) and MASK (the mask corresponding of stain on cloth).

# Type of synthethic data generated: 

- Type 1: circle deformed and resized (c.f. code : deform_circle_add_V2 = True)

- Type 2 : circle add multiples little spikes at border (c.f. code : deform_circle_add_V3 = True) 

- Type 3 : random shape controlled by edginess and radius (c.f. code : random_shape_edgy_add = True)

# PART 2 : Object detection model with Yolo v5 

Here we choose first implement yolov5 model (c.f. https://github.com/ultralytics/yolov5) for the identification with bounding boxes of stain on images. We define one class (with stain) as 0, followed by the training process...
The training and validation were done on google colab platform, c.f. https://colab.research.google.com/drive/1ePqvXijYftsFizAHqdT6Sv1_PP9IwAEh#scrollTo=plhfUHXZyvkM

Dataset for training, training results, testset inferences results of different versions stored in zip format on google drive: Project_Cetia_Thang_Thai_Khoi > dataset_stain_insa-cetia
