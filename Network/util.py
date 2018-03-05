# Data Testing, makes image label pairs in reasonable resolutions.
from filetools import parse_dir
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import cv2
import tensorflow as tf
import ops

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as tfcheckprint

flags = tf.app.flags
FLAGS = flags.FLAGS

def get_Im_Specs():
  imgW    = FLAGS.imgW
  imgH    = FLAGS.imgH
  patW    = FLAGS.patch_size
  patH    = FLAGS.patch_size
  num_pat = imgW * imgH / (patW * patH)
  return int(imgW),int(imgH),int(patW),int(patH),int(num_pat)

def ckpt_inspect(file):
  tfcheckprint(file_name = file, tensor_name = '', all_tensors = True,all_tensor_names = True)
  input("PRESS ENTER TO CONTINUE...")

def tfrecord_inspect(file):
  c = 0
  for record in tf.python_io.tf_record_iterator(file):
    c += 1
  print("%d records in %s"%(c,file))
  input("PRESS ENTER TO CONTINUE...")

def tfrecord_count_records(file):
  c = 0
  for record in tf.python_io.tf_record_iterator(file):
    c += 1
  return c


def tfrecord_advanced_inspect(file):
  record_iterator = tf.python_io.tf_record_iterator(path=file)

  for string_record in record_iterator:
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Get the features you stored (change to match your tfrecord writing code)
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    plant =      (example.features.feature['p_label_r']
                                  .bytes_list
                                  .value[0])
    disease =    (example.features.feature['d_label_r']
                                  .bytes_list
                                  .value[0])
    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    plant = np.fromstring(plant, dtype=np.uint8)
    disease = np.fromstring(disease, dtype=np.uint8)
    # Print the image shape; does it match your expectations?
    print(img_1d.shape)
    print(plant.shape)
    print(disease.shape)

def Image_To_Patch(image):
  with tf.variable_scope("Image_To_Patch") as scope:
    imgW,imgH,patW,patH,numP = get_Im_Specs()
    chan = image.shape[-1].value
    image = tf.squeeze(image)
    image = tf.reshape(image,[1,imgH,imgW,chan])
    patSize = [1,patH,patW,1]

    patches      = tf.extract_image_patches(image,patSize,patSize,[1,1,1,1],'VALID')
    patches      = tf.reshape(patches,[-1,patH,patW,chan])
    # tf.summary.image('Patches',patches)
    return patches

def ImSizeToPatSize(image):
  blkH = int(100)
  blkW = int(100)
  block_shape = (blkH,blkW)
  return block_shape

def Patch_To_Image(patch):
  with tf.variable_scope("Patch_To_Image") as scope:
    imgW,imgH,patW,patH,numP = get_Im_Specs()
    patch = tf.squeeze(patch)
    chan = tf.minimum(1,patch.shape[-1].value)
    imgSize = [1,imgH,imgW,chan]
    patSize = [-1,patH,patW,chan]
    patch = tf.reshape(patch,patSize)

    img_re = tf.reshape  (patch, [int(imgH / patH), int(imgW / patW), patH, patW])
    img_tr = tf.transpose(img_re,[0,2,1,3])
    image  = tf.reshape  (img_tr,imgSize)

    # tf.summary.image('P->I',image)
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
      lab = tf.reshape(label[x],[tf.size(label[x])])
      # tf.summary.histogram('Pat_%d'%x,lab)
      y,idx = tf.unique(lab)
      disc_label.append(tf.minimum(1,tf.size(y)-1))
    disc_label = tf.stack(disc_label)
    # disc_label = tf.one_hot(disc_label,2)
    disc_label = tf.reshape(disc_label,(disc_label.shape[0].value,1,1))
    return disc_label


# Tape GT saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def imsave(im_bat, name = 'Image_Save',int_scale = True, stride = True):
  with tf.variable_scope(name) as scope:
    im_bat = tf.cast(im_bat,tf.float32)
    if int_scale:
      intensity = tf.convert_to_tensor(255,dtype = tf.float32,name = 'intensity')
      im_bat    = tf.scalar_mul(intensity,im_bat)
    im_bat      = tf.reshape(im_bat,[FLAGS.batch_size,FLAGS.imgH,FLAGS.imgW,-1])
    im_bat      = tf.cast(im_bat,tf.uint8)
    save_ims    = []
    for i in range(im_bat.shape[0].value):
      save_img = tf.image.encode_png(im_bat[i,:,:])
      save_ims.append(save_img)
    if stride:
      im_bat = im_bat[:,::FLAGS.save_stride,::FLAGS.save_stride,:]
    if FLAGS.log_imgs:
      tf.summary.image(name,im_bat)
    save_ims_grouped  = tf.tuple(save_ims)
    return save_ims_grouped

def get_params():
  num_param = 0
  for var in tf.trainable_variables():
    for dim in var.get_shape():
      num_param = num_param + dim.value
  return num_param

def showimgs(img,lab):
  f, (img_p, lab_p) = plt.subplots(2)
  img_p.imshow(img)
  lab_p.imshow(lab)
  plt.show()

def squish_to_batch(net):
  batch, input_height, input_width, channels = net.get_shape().as_list()
  net = tf.reshape(net,[batch,input_height*input_width*channels])
  return net

# print(s_factors(100))
