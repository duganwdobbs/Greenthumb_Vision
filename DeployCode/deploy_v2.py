import tensorflow as     tf
import numpy      as     np

import platform
import os

import ops
import util

import resnet_modules
resnetA = resnet_modules.block35
resnetB = resnet_modules.block17
resnetC = resnet_modules.block8

# Limiting the amount of logging that gets spewed to the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setting up parser flags
flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'E:/Greenthumb_Vision'          ,'Base os specific DIR')
  flags.DEFINE_string ('log_dir'   ,'F:/Greenthumb_Vision'          ,'Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')
  flags.DEFINE_string ('log_dir'  ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss'            ,True  ,'If we use l2 regularization')
flags.DEFINE_boolean('batch_norm'         ,True  ,'If we use batch normalization')
flags.DEFINE_boolean('advanced_logging'   ,False ,'If we log metadata and histograms')
flags.DEFINE_boolean('log_imgs'           ,False ,'If we log images to tfrecord')


flags.DEFINE_integer('num_epochs'         ,1     ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size'         ,4     ,'Batch size for training.')
flags.DEFINE_integer('train_steps'        ,100000,'Number of steps for training on counting')
flags.DEFINE_integer('num_plant_classes'  ,9    ,'# Classes')
flags.DEFINE_integer('num_disease_classes',10    ,'# Classes')


flags.DEFINE_string ('run_dir'    , FLAGS.log_dir + '/network_log/','Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  , FLAGS.base_dir                 ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  ,'greenthumb.ckpt'               ,'Checkpoint name')
flags.DEFINE_string ('net_name'   ,'PlantVision'                   ,'Network name')

# The Classification network.
# The Classification network.
def inference(images,training,name,trainable = True):
  ops.init_scope_vars()
  with tf.variable_scope(name) as scope:
    plants   = [ 'Apple','Cherry','Corn','Grape','Peach','Strawberry','Bell_Pepper','Potato','Tomato']
    # Preform some image preprocessing
    net = images
    net = ops.delist(net)
    net = tf.cast(net,tf.float32)
    net = ops.im_norm(net)

    # If we receive image channels in a format that shouldn't be normalized,
    #   that goes here.

    with tf.variable_scope('Process_Network'):
      # Run two dense reduction modules to reduce the number of parameters in
      #   the network and for low parameter feature extraction.
      net = ops.dense_reduction(net,training,filters = 4 , kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_1')
      net = ops.dense_reduction(net,training,filters = 12, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_2')
      net = ops.dense_reduction(net,training,filters = 16, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_3')

      # Run the network over some resnet modules, including reduction
      #   modules in order to further reduce the parameters and have a powerful,
      #   proven network architecture.
      net = resnetA(net)
      net = resnetA(net)
      net = ops.dense_reduction(net,training,filters = 14, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Redux_4')

      net = resnetA(net)
      net = resnetA(net)
      net = ops.dense_reduction(net,training,filters = 18, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Redux_5')


      net = resnetB(net)
      net = resnetB(net)
      net = ops.dense_reduction(net,training,filters = 22, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Redux_6')


      net = resnetC(net)
      # net = resnetC(net)
      # net = ops.dense_reduction(net,training,filters = 56, kernel = 3, stride = 2,
      #                           activation=tf.nn.leaky_relu,trainable=trainable,
      #                           name = 'Dense_Block_7')

      net = util.squish_to_batch(net)
      _b,neurons = net.get_shape().as_list()

    with tf.variable_scope('Plant_Neurons') as scope:
      p_log = tf.layers.dense(net,neurons,name = 'Plant_Neurons')
      p_log = tf.layers.dense(p_log,FLAGS.num_plant_classes,name = 'Plant_Decider')
      p_log = tf.squeeze(p_log)
    with tf.variable_scope('Disease_Neurons') as scope:
      # Construct a number of final layers for diseases equal to the number of plants.
      d_log = []
      chan = tf.layers.dense(net,neurons, name = 'Disease_Neurons')
      for x in range(FLAGS.num_plant_classes):
        d_n = tf.layers.dense(chan,FLAGS.num_disease_classes,name = '%s_Decider'%plants[x])
        d_n = tf.squeeze(d_n)
        d_log.append(d_n)
      with tf.variable_scope('DieaseFormatting') as scope:
        d_log = tf.stack(d_log)

    return p_log,d_log

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
      saver.restore(self.sess,'./model.ckpt')
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
    p_conf = p_logs[p_log] * 100 * .95
    d_conf = d_logs[d_log] * 100 * .85
    plant  = self.p_log_to_desc(p_log)
    disease= self.d_log_to_desc(p_log,d_log)
    print("We are %.2f%% certain that this is a %s plant, and %.2f%% certain that this %s has %s disease."%(p_conf,plant,d_conf,plant,disease))


def main(_):
  net = Deploy_Network


if __name__ == '__main__':
  tf.app.run()
