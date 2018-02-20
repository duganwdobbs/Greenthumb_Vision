# Data Testing, makes image label pairs in reasonable resolutions.
from filetools import parse_dir
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import cv2
import tensorflow as tf
import ops

flags = tf.app.flags
FLAGS = flags.FLAGS

def Image_To_Patch(image):
  with tf.variable_scope("Image_To_Patch") as scope:
    patch_shape = (int(FLAGS.imgH / FLAGS.patch_size),int(FLAGS.imgW / FLAGS.patch_size))
    paddn_shape = (2,2)
    padding     = tf.zeros(shape = paddn_shape,dtype = tf.int32)
    patch = tf.space_to_batch_nd(image,patch_shape,padding)
    return patch

def ImSizeToPatSize(image):
  blkH = int(100)
  blkW = int(100)
  block_shape = (blkH,blkW)
  return block_shape

def Patch_To_Image(patch):
  with tf.variable_scope("Patch_To_Image") as scope:
    image_shape = (int(FLAGS.imgH / FLAGS.patch_size),int(FLAGS.imgW / FLAGS.patch_size))
    crops       = [ [0 ,0], [0, 0] ]
    image       = tf.batch_to_space_nd(patch,image_shape,crops)
    return image

def s_factors(n):
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n > 1:
    factors.append(n)
  return factors

def factors(a,b):
  a = s_factors(a)
  b = s_factors(b)
  c = []
  for x in range(len(a)):
    if a[x] in b:
      c.append(a[x])
      b.remove(a[x])
  return c

# Generates True / False labels in the shape of [#Batch][2], where 0 is false,
# and 1 is true.
def disc_label_gen(label):
  with tf.variable_scope('Discriminator_Label_Gen') as scope:
    disc_label = []
    for x in range(label.shape[0].value):
      y,idx = tf.unique(label[x])
      disc_label.append(tf.minimum(1,tf.size(y)-1))
    disc_label = tf.stack(disc_label)
    # disc_label = tf.one_hot(disc_label,2)
    disc_label = tf.reshape(disc_label,(disc_label.shape[0].value,1,1))
    return disc_label


# Tape GT saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def imsave(im_bat,int_scale = True, name = 'Image_Save'):
  with tf.variable_scope(name) as scope:
    if int_scale:
      intensity = tf.convert_to_tensor(255,dtype = tf.float32,name = 'intensity')
      im_bat    = tf.scalar_mul(intensity,im_bat)
    im_bat      = tf.cast(im_bat,tf.uint8)
    save_ims    = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(im_bat[i,:,:])
      save_ims.append(save_img)

    tf.summary.image(name,im_bat[:,::FLAGS.save_stride,::FLAGS.save_stride,:])
    save_ims_grouped  = tf.tuple(save_ims)
    return save_ims_grouped

def showimgs(img,lab):
  f, (img_p, lab_p) = plt.subplots(2)
  img_p.imshow(img)
  lab_p.imshow(lab)
  plt.show()


# print(s_factors(100))
