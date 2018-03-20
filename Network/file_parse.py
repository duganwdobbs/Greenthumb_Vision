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
  base_dir = 'E:/plantvillage/color'
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
            ['Healthy','Scab','Black Rot','Cedar Apple Rust'],                      # Apple
            ['Healthy','Powdery Mildew'],                                           # Cherry
            ['Healthy','Cercospora Leaf Spot','Common Rust','Northern Leaf Blight'],# Corn
            ['Healthy','Black Rot','Esca (Black Measles)','Leaf Blight'],           # Grape
            ['Healthy','Bacterial Spot'],                                           # Peach
            ['Healthy','Leaf Scorch'],                                              # Strawbery
            ['Healthy','Bacterial Spot'],                                           # Bell Pepper
            ['Healthy','Early Blight','Late Blight,Leaf Scorch'],                   # Potato
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

examples = []

save_dir = base_dir + '../../'
img_ext = '.JPG'

plant_test  = []
plant_train = []
for x in range(len(plants)):
  disease_test  = []
  disease_train = []
  for y in range(len(diseases[x])):
    # print("%s %s"%(plants[x],diseases[x][y]))
    # Find the files that conform to given plant cat X and disease cat Y
    file_list = filetools.find_files(base_dir + '/' + director[x][y] + '/',img_ext)
    # Create image / label tuples
    img_lab_l = [(base_dir + '/' + director[x][y] + '/' + f, x, y) for f in file_list]
    # print("%s %s has %d examples..."%(plants[x],diseases[x][y],len(file_list)))
    # Determine the number of testing examples
    test_len  = len(file_list) * 3 // 10
    # Create a test list
    test_list = img_lab_l[         :test_len]
    # Create a train list starting from the end of the test list
    train_list= img_lab_l[test_len:         ]
    # Append these lists to the running plant list.
    print("%s %s: %d Testing, %d Training"%(plants[x],diseases[x][y],len(test_list),len(train_list)))
    for f in test_list:
      disease_test.append(f)
    for f in train_list:
      disease_train.append(f)
    # End disease loop

  if plants[x] == 'Corn':
    print("Not training on corn.")
  else:
    # Write the disease class test list.
    print("%s: %d Testing, %d Training"%(plants[x],len(disease_test),len(disease_train)))
    shuffle(disease_test)
    # writer(disease_test,save_dir + '%s-Test.tfrecords'%plants[x])
    # Write the disease class train list.
    shuffle(disease_train)
    # writer(disease_train,save_dir + '%s-Train.tfrecords'%plants[x])

    #Append to the running lists for plant test+train
    for f in disease_test:
      plant_test.append(f)
    for f in disease_train:
      plant_train.append(f)

print("%s: %d Testing, %d Training"%("Full Data Split",len(plant_test),len(plant_train)))
# Write the plant test list
shuffle(plant_test)
# writer(plant_test,save_dir  + 'PlantVision-Test.tfrecords')
# Write the plant train list
shuffle(plant_train)
# writer(plant_train,save_dir + 'PlantVision-Train.tfrecords')
