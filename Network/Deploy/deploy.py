import tensorflow as     tf
import numpy      as     np

import platform
import os

import ops
import util

# Limiting the amount of logging that gets spewed to the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setting up parser flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# The Classification network.
def inference(images,training,name,trainable = True):
  ops.init_scope_vars()
  with tf.variable_scope(name) as scope:
    # Preform some image preprocessing
    net = images
    net = ops.delist(net)
    net = tf.cast(net,tf.float32)
    net = ops.batch_norm(net,training,trainable)
    # net = tf.fft2d(net)
    # If we receive image channels in a format that shouldn't be normalized,
    #   that goes here.
    with tf.variable_scope('Process_Network'):
      # Run two dense reduction modules to reduce the number of parameters in
      #   the network and for low parameter feature extraction.
      net = ops.dense_reduction(net,training,filters = 4, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_1')
      net = ops.dense_reduction(net,training,filters = 8, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_2')
      net = ops.dense_reduction(net,training,filters = 4, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_3')
      net = ops.dense_reduction(net,training,filters = 8, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_4')

      # Run the network over some resnet modules, including reduction
      #   modules in order to further reduce the parameters and have a powerful,
      #   proven network architecture.
      net = ops.inception_block_a(net,training,trainable,name='inception_block_a_1')
      net = ops.dense_reduction(net,training,filters =16, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_5')

      net = ops.inception_block_a(net,training,trainable,name='inception_block_a_2')
      net = ops.inception_block_a(net,training,trainable,name='inception_block_a_3')
      net = ops.dense_reduction(net,training,filters =24, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_6')

      net = ops.inception_block_b(net,training,trainable,name='inception_block_b_1')
      net = ops.inception_block_b(net,training,trainable,name='inception_block_b_2')

      # Commenting out for proof of concept
      # net = ops.inception_block_c(net,training,trainable,name='inception_block_c_1')
      # net = ops.inception_block_c(net,training,trainable,name='inception_block_c_1')

      # We're leaving the frequency domain in order to preform fully connected
      #    inferences. Fully connected inferences do not work with imaginary
      #    numbers. The error would always have an i/j term that will not be able
      #    to generate a correct gradient for.

      # net = tf.ifft2d(net)

      # Theoretically, the network will be 8x8x128, for 8192 neurons in the first
      #    fully connected network.
      net = util.squish_to_batch(net)
      _b,neurons = net.get_shape().as_list()

    with tf.variable_scope('Plant_Neurons') as scope:

      # Fully connected network with number of neurons equal to the number of
      #    parameters in the network at this point
      p_log = tf.layers.dense(net,neurons,name = 'Plant_Neurons')

      # Fully connected layer to extract the plant classification
      #    NOTE: This plant classification will be used to extract the proper
      #          disease classification matrix
      p_log = tf.layers.dense(p_log,FLAGS.num_plant_classes,name = 'Plant_Decider')

    with tf.variable_scope('Disease_Neurons') as scope:
      # Construct a number of final layers for diseases equal to the number of
      # plants.
      d_net = []
      chan = tf.layers.dense(net,neurons)#, name = 'Disease_Neurons)'
      for x in range(FLAGS.num_plant_classes):
        d_n = tf.layers.dense(chan,FLAGS.num_disease_classes,name = 'Disease_%d_Decider'%x)
        d_net.append(d_n)
      d_net = tf.stack(d_net)
      d_log = d_net

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
