import tensorflow as tf
import PlantNet   as pn

import ops
import util

# Limiting the amount of logging that gets spewed to the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setting up parser flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# Defining parser flags (Fancy global variables)
if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'E:/Greenthumb_Vision'           ,'Base os specific DIR')
  flags.DEFINE_string ('log_dir'   ,'F:/Greenthumb_Vision'           ,'Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')
  flags.DEFINE_string ('log_dir'   ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')

flags.DEFINE_integer('batch_size'         ,1     ,'Batch size for training.')
flags.DEFINE_integer('num_plant_classes'  ,10    ,'# Classes')
flags.DEFINE_integer('num_disease_classes',10    ,'# Classes')

flags.DEFINE_boolean('batch_norm'         ,True  ,'If we use batch normalization')

flags.DEFINE_string ('run_dir'    , FLAGS.log_dir + '/network_log/','Location to store the Tensorboard Output')

class Deploy_Network:

  def init(self):
    with tf.Graph().as_default():
      # Allowing the network to use a non-default GPU if necessary.
      config                          = tf.ConfigProto(allow_soft_placement = True)

      # Making it so that TensorFlow doesn't eat all of the video ram.
      config.gpu_options.allow_growth = True

      # Setting up our session, saved as a class var so that we can use it later
      self.sess                       = tf.Session(config = config)

      # Defining our image dimensions, we will use this placeholder to send
      #   data through the network
      self.images                     = tf.placeholder(tf.float32,(256,256))

      # Getting the outputs from the network. These will be matrices in the
      #  shape of:
      #    p_log: [# Plant Classes #]
      #    d_log: [# Plant Classes #][#Disease Classes#]
      self.p_log, self.d_log          = pn.inference(images,training = False,name = 'PlantVision',trainable = False)

      # Setting up system to load a saved model
      saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
      # Loading the saved model
      saver.restore(sess,tf.train.latest_checkpoint(FLAGS.run_dir))

  # Receives an image
  # Returns output of statistical probability matrix where the max index is
  #   the label.
  def run(self,feed_image):
    outputs = [self.p_log,self.d_log]
    _p_log,_d_log = self.sess.run(outputs,{self.images: feed_image})
    return _p_log, _d_log
