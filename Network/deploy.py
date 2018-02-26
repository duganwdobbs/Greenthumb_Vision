import tensorflow as     tf
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
      self.p_log, self.d_log          = inference(im,training = False,name = 'PlantVision',trainable = False)
      self.p_logs = tf.squeeze(self.p_log)
      self.d_logs = tf.squeeze(self.d_log)

      # Setting up system to load a saved model
      saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      # Loading the saved model
      print("Network defined, loading from checkpoint...")
      print(tf.train.latest_checkpoint(FLAGS.run_dir))
      saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.run_dir))
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

  def get_plants(self):
    return self.plants

  def d_log_to_desc(self,p_log,d_log):
    return self.diseases[p_log][d_log]

  def get_diseases(self):
    return self.diseases

def main(_):
  net = Deploy_Network


if __name__ == '__main__':
  tf.app.run()
