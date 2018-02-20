# Data Testing
from filetools import parse_dir
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
#Images

def fix_aspect(imgs,labs):
  for x in range(len(imgs)):
    img = Image.open(imgs[x])
    img = np.asarray(img)
    lab = Image.open(labs[x])
    newshape = (img.shape[1],img.shape[0])
    lab = lab.resize(newshape)
    lab.save(labs[x].replace(lab_ext,new_ext))

def showimgs(img,lab):
  f, (img_p, lab_p) = plt.subplots(2)
  img_p.imshow(img)
  lab_p.imshow(lab)
  plt.show()

def gt_to_png(lab):
  fname = lab
  lab = Image.open(lab)
  lab = np.asarray(lab)
  lab.flags.writeable = True
  lab = lab[:,:]
  thresh = 145
  lab[np.where(lab ==  1)]  = 255
  lab = Image.fromarray(lab)
  lab.save(fname.replace('.jpg','.png'))

def make_gt(img):
  f_name = img
  img = Image.open(img)
  img = np.asarray(img)
  img.flags.writeable = True
  kernel = np.array( [[ 1, 1],
                      [ 1, 1] ] )
  kernel = np.ones((3,3))
  #dilate
  img = cv2.dilate(img,kernel)
  #erode
  img = cv2.erode(img,kernel)
  plt.imshow(img)
  plt.show()

  #


data_dir = 'E:/Cows/Cows Found'

img_ext = '.JPG'
lab_ext = 'O.jpg'

imgs = parse_dir(data_dir,img_ext)

print(imgs)

for img_ in imgs:
  for img in img_:
    print(img)
    make_gt(img)
