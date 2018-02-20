from PIL import Image
import numpy as np
import tensorflow as tf
import os.path
import random

# Basic model parameters as external flags.
imgW = int(256)
imgH = int(256)

def sizes():
  return imgW,imgH

flags = tf.app.flags
FLAGS = flags.FLAGS

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE =
VALIDATION_FILE =

# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_example(img_raw,p_label,d_label):
  example = tf.train.Example(features=tf.train.Features(feature={
      'image_raw': _bytes_feature(img_raw),
      'p_label_r': _bytes_feature(p_label),
      'd_label_r': _bytes_feature(d_label)  }))
  return example

def write_image_label_pairs_to_tfrecord(filename_pairs, tfrecords_filename):
    """Writes given image/label pairs to the tfrecords file.
    The function reads each image/label pair given filenames
    of image and respective label and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, label_filepath)
        Array of tuples of image/label filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/label pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    print(tfrecords_filename)
    i = 0
    for img_path, lab_path, dat_path in filename_pairs:

        img = Image.open(img_path)
        img = img.resize((imgW,imgH),Image.NEAREST)
        img = np.asarray(img)

        p_label =
        d_label =

        count = 0
        with open(dat_path,'r') as cnt_file:
          count = np.asarray(cnt_file.read())

        img       =     img.astype(np.uint8)
        p_label   = p_label.astype(np.uint8)
        d_label   = d_label.astype(np.uint8)

        img_raw   =     img.tobytes()
        p_label_r = p_label.tobytes()
        d_label_r = d_label.tobytes()

        example = get_example(img_raw,p_label_r,d_label_r)
        writer.write(example.SerializeToString())

        img_raw = np.fliplr(img).astype(np.uint8).tobytes()
        example = get_example(img_raw,p_label_r,d_label_r)
        writer.write(example.SerializeToString())

        img_raw   =   np.flipud(img).astype(np.uint8).tobytes()
        example = get_example(img_raw,p_label_r,d_label_r)
        writer.write(example.SerializeToString())

        img_raw   =   np.fliplr(np.flipud(img)).astype(np.uint8).tobytes()
        example = get_example(img_raw,p_label_r,d_label_r)
        writer.write(example.SerializeToString())

        i  = i + 4
        print("Processed " + str(i) + " images...")
    print("Done!")

    writer.close()

def read_decode(tfrecord_filenames_queue, num_classes):
    """Return image/label tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/label tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    Returns
    -------
    image, label : tuple of tf.int32 (image, label)
        Tuple of image/label tensors
    """

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
        'count_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['p_label_r'], tf.uint8)
    count = tf.decode_raw(features['d_label_r'], tf.uint8)

    image_shape = tf.stack([imgH, imgW, 3])

    image = tf.reshape(image, image_shape)
    image = tf.cast(image,tf.float32)
    image = tf.divide(image,255)

    count = tf.reshape(count, [1])

    return image, label, count

def inputs(train):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, FLAGS.FLAGS.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  filename = os.path.join(FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
  num_epochs = FLAGS.num_epochs if train else 1
  print('Input file: ' + filename)
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    image, label, count = read_decode(filename_queue,FLAGS.num_classes)

    if train:
        images, p_label, d_label = tf.train.shuffle_batch(
            [image, p_label, d_label], batch_size=FLAGS.batch_size, num_threads=1,
            capacity=10 + 2 * FLAGS.batch_size,
            min_after_dequeue=10)
    else:
        images, p_label, d_label = tf.train.batch(
            [image, p_label, d_label], batch_size=FLAGS.batch_size, num_threads=2,
            capacity=10 + 2 * FLAGS.batch_size)
    return images, p_label, d_label
