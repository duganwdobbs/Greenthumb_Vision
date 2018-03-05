import tensorflow as     tf
import numpy      as     np
from   PlantNet   import inference

import platform
import os

import ops
import util

# Limiting the amount of logging that gets spewed to the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setting up parser flags
flags = tf.app.flags
FLAGS = flags.FLAGS
class Deploy_Network:

  def __init__(self):
    with tf.Graph().as_default():
      FLAGS.batch_size = 1

      self.plants   = [ 'Apple','Cherry','Corn','Grape','Peach','Strawberry','Bell Pepper','Potato','Tomato']
      self.diseases = [
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
      print("Building network model...")

      # Allowing the network to use a non-default GPU if necessary.
      config                          = tf.ConfigProto(allow_soft_placement = True)

      # Making it so that TensorFlow doesn't eat all of the video ram.
      config.gpu_options.allow_growth = True

      # Setting up our session, saved as a class var so that we can use it later
      self.sess                       = tf.Session(config = config)

      # Defining our image dimensions, we will use this placeholder to send
      #   data through the network
      self.ims                     = tf.placeholder(tf.float32,(256,256,3))

      # Getting the outputs from the network. These will be matrices in the
      #  shape of:
      #    p_log: [# Plant Classes #]
      #    d_log: [# Plant Classes #][#Disease Classes#]
      im = tf.reshape(self.ims,[1,256,256,3])
      self.p_log, self.d_log          = inference(im,training = True,name = 'PlantVision',trainable = False)

      self.p_logs = tf.squeeze(self.p_log)
      self.p_log  = tf.argmax(self.p_logs)

      self.d_logs = tf.squeeze(self.d_log)

      size = [1,FLAGS.num_disease_classes]
      # Extract the disease logit per example in batch
      index = self.p_log
      d_log = []
      start = [index,0]
      d_log = tf.slice(self.d_logs,start,size)
      self.d_logs = tf.reshape(d_log,[FLAGS.batch_size,FLAGS.num_disease_classes])

      self.p_logs = tf.nn.softmax(self.p_logs)
      self.d_logs = tf.nn.softmax(self.d_logs)

      # Setting up system to load a saved model
      saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      # Loading the saved model
      print("Network defined, loading from checkpoint...")
      saver.restore(self.sess,FLAGS.run_dir + 'model.ckpt')
      print("Network Loaded from checkpoint.")
  # end __init__

  def debug(self):
    vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    for v in vars:
      print(v,end="")

  # Receives an image
  # Returns output of statistical probability matrix where the max index is
  #   the label.
  def run(self,feed_image):
    outputs = [self.p_logs,self.d_logs]
    _p_logs,_d_logs = self.sess.run(outputs,{self.ims: feed_image})
    return _p_logs, _d_logs
  # end run

  def p_log_to_desc(self,p_log):
    return self.plants[p_log]

  def d_log_to_desc(self,p_log,d_log):
    return self.diseases[p_log][d_log]

  def result_verbose(self,p_logs,d_logs):
    p_logs = np.squeeze(p_logs)
    d_logs = np.squeeze(d_logs)
    p_log  = p_logs
    p_log  = np.argmax(p_log)
    d_log  = d_logs
    d_log  = np.argmax(d_log)
    p_conf = p_logs[p_log] * 100
    d_conf = d_logs[d_log] * 100
    plant  = self.p_log_to_desc(p_log)
    disease= self.d_log_to_desc(p_log,d_log)
    print("We are %.2f%% certain that this is a %s plant, and %.2f%% certain that this %s has %s disease."%(p_conf,plant,d_conf,plant,disease))


def main(_):
  net = Deploy_Network


if __name__ == '__main__':
  tf.app.run()
