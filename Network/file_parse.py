# AmazonParse.py
import filetools
import os
import numpy as np
from tfrecord import write_image_label_pairs_to_tfrecord as writer
from random import shuffle
import platform
from PIL import Image

# Function for parsing and making flood TFRecord file. Automatically pairs and
# builds image / label pairs.
def data_pair(flist,direct,filename):
  gud_list = []
  for f in flist:
    if os.path.isfile(direct + f.replace(label_ext,image_ext)):
      gud_list.append((direct + f.replace(label_ext,image_ext),direct + f))
  print(flist)
  print(gud_list)
  print(len(flist))
  writer(gud_list, direct + filename)


# Where to put your base directories.
base_dir = ''
if   platform.system() == 'Windows':
  base_dir = 'D:/Cows/CowsFound'
elif platform.system() == 'Linux':
  base_dir = '/data0/ddobbs/Cows/Cows Found'
  # /home/ddobbs/BinaLab-Semantic-Segmentation/data

save_dir = base_dir + '../../'
img_ext = '.JPG'
sec_ext = '_2.jpg'
lab_ext = '.png'
cnt_ext = '.dat'
# Build the list of JSON files
img_list = filetools.find_files(base_dir,sec_ext)
lab_list = filetools.find_files(base_dir,lab_ext)
dat_list = filetools.find_files(base_dir,cnt_ext)

# for img_path in img_list:
#   img = Image.open(img_path)
#   img = img.resize((4000,3000),Image.NEAREST)
#   img.save(img_path.replace(img_ext,sec_ext))


newlist = []
for x in range(len(img_list)):
  newlist.append((img_list[x],lab_list[x],dat_list[x]))

shuffle(newlist)

# for im in img_list:
#   print(im)
#   with open(im.replace(img_ext,cnt_ext),'w+') as datfile:
#     print('How many cows in ',end = '')
#     x = input(im)
#     datfile.write(x)

print(newlist)
# input('ctrl c here.')

test_list  = []
train_list = []

for x in range(len(img_list)):
  if x > len(img_list) * 3 / 10:
    train_list.append(newlist[x])
  else:
    test_list.append(newlist[x])


# Build the TFRecord files.
writer(train_list,base_dir+'train_cows.tfrecords')
writer(test_list,base_dir+'test_cows.tfrecords')
