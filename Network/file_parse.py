# AmazonParse.py
import filetools
import os
import numpy as np
from tfrecord import write_image_label_pairs_to_tfrecord as writer
from random import shuffle
import platform
from PIL import Image


# Where to put your base directories.
base_dir = ''
if   platform.system() == 'Windows':
  base_dir = 'D:/Greenthumb_Vision/color'
elif platform.system() == 'Linux':
  base_dir = '/data0/ddobbs/Cows/Cows Found'
  # /home/ddobbs/BinaLab-Semantic-Segmentation/data

# 9 plants     10 Max Diseases
# Apple       : Healthy : Scab : Black Rot : Cedar Apple Rust
# Cherry      : Healthy : Powdery Mildew
# Corn        : Healthy : Cercospora Leaf Spot : Common Rust : Northern Leaf Blight
# Grape       : Healthy : Black Rot : Esca (Black Measles) : Leaf Blight
# Peach       : Healthy : Bacterial Spot
# Strawberry  : Healthy : Leaf Scorch
# Bell Pepper : Healthy : Bacterial Spot
# Potato      : Healthy : Early Blight : Late Blight : leaf Scorch
# Tomato      : Healthy : Bacterial Spot : Early Blight : Late Blight: Leaf Mold : Septoria Leaf Spot : Spider Mite : Target Spot : Mosaic Virus : Yellow Leaf Curl Virus

plants   = [ 'Apple','Cherry','Corn','Grape','Peach','Strawberry','Bell Pepper','Potato','Tomato']
diseases = [
            ['Healthy','Scab','Black Rot','Cedar Apple Rust'],
            ['Healthy','Powdery Mildew'],
            ['Healthy','Cercospora Leaf Spot','Common Rust','Northern Leaf Blight'],
            ['Healthy','Black Rot,Esca (Black Measles)','Leaf Blight'],
            ['Healthy','Bacterial Spot'],
            ['Healthy','Leaf Scorch'],
            ['Healthy','Bacterial Spot'],
            ['Healthy','Early Blight','Late Blight,Leaf Scorch'],
            ['Healthy','Bacterial Spot','Early Blight','Late Blight','Leaf Mold','Septoria Leaf Spot','Spider Mite','Target Spot','Mosaic Virus','Yellow Leaf Curl Virus']
              ]
director = [
            ['Apple___healthy','Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust'],
            ['Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew'],
            ['Corn_(maize)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight'],
            ['Grape___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'],
            ['Peach___healthy','Peach___Bacterial_spot'],
            ['Strawberry___healthy','Strawberry___Leaf_scorch'],
            ['Pepper,_bell___healthy','Pepper,_bell___Bacterial_spot'],
            ['Potato___healthy','Potato___Early_blight','Potato___Late_blight'],
            ['Tomato___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']
              ]

save_dir = base_dir + '../../'
img_ext = '.JPG'
img_lab = []
# Loop through all of the plants
for x in range(len(plants)):
  # Loop through all of the dieases for this plant
  for y in range(len(diseases[x])):
    # Build a list of images from this data
    file_list = filetools.find_files(base_dir + '/' + director[x][y] + '/',img_ext)
    print(len(file_list))
    for z in file_list:
      # Create an image / label triplet
      img_lab.append((base_dir + '/' + director[x][y] + '/' + z,x,y))

# Shuffle the list of tuples
shuffle(img_lab)

# Examples
ex = len(img_lab)
print("Found %d examples."%ex)
test  = ex // 7
train = ex // 3

writer(img_lab[:train],save_dir + 'PlantVision-Test.tfrecords')
writer(img_lab[train:],save_dir + 'PlantVision-Train.tfrecords')
