import tensorflow as tf
import ops
import util

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('bn_scope'       , 0 , 'Variable for BN Scope')
flags.DEFINE_integer('spec_conv_scope', 0 , 'Variable for Conv2d Scope')

def init():
  FLAGS.bn_scope        = 0
  FLAGS.spec_conv_scope = 0

# Utility function to get shape of a 4 dim network
def get_shape(source):
  batch, input_height, input_width, channels = source.get_shape().as_list()
  return batch, input_height, input_width, channels

# Creates a filter for a spectral convolution
def create_filter(filters,net):
  batch, height, width, channels = source.get_shape().as_list()
  r = tf.truncated_normal([channels,filters,height,width],mean=0,stddev=.01)
  i = tf.truncated_normal([channels,filters,height,width],mean=0,stddev=.01)
  filt = tf.conplex(r,i)
  return filt

# Creates biases for a spectral convolution
def create_bias(filters):
  r = tf.truncated_normal([filters],mean=0,stddev=.01)
  i = tf.truncated_normal([filters],mean=0,stddev=.01)
  bias = tf.complex(r,i)
  return bias

# Spectral conv2d, assumes net comes in as a representation of a FFT
def spec_conv2d(net, training, filters, kernel = 3, stride = 1,
                activation = ops.lrelu, trainable = True, name = None):
  if name is None:
    name = "spec_conv2d_%d"%FLAGS.spec_conv_scope
  if net.dtype is not tf.complex64:
    raise Exception("Spectral Conv2d received image not in complex64")
  # Need to define how to reuse
  with tf.variable_scope(name) as scope:
    filt = create_filter(net,)
           # TODO: Note, dilation rate must be maintained
           # TODO: Note, function to create vars must work with reuse and trainable
    b    = create_bias(filtrs)
    net  = net * filt + b
    net  = ops.batch_norm(net,training,trainable,ops.lrelu)
    if stride is not 1:
      net = spec_pool(net,stride)
    return

# Spectral Pooling (Not re-padding)
# Takes the current network and cuts off the outer edges
def spec_pool(net,stride):
  batch, input_height, input_width, channels = get_shape(net)
  new_height = input_height // stride
  trim_h     = (input_height - new_height) / 2
  new_width  = input_width  // stride
  trim_w     = (input_width - new_width) / 2
  net = net[:,trim_h:trim_h+new_height,trim_w:trim_w + new_width,:]
  return net


#         '''~~~~ FROM HERE ON, ADVANCED OPERATIONS ~~~~'''

def dense_reduction(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = ops.lrelu, trainable = True,name = 'Dense_Block'):
  with tf.variable_scope(name) as scope:
    net = delist(net)
    for n in range(kmap):
      out = spec_conv2d(net, training, filters=filters, kernel=kernel, stride=1,
                        activation=activation, trainable=trainable, name = '_map_%d'%n)
      net = tf.concat([net,out],-1,name = '%d_concat'%n)
    if stride is not 1:
      net = spec_pool(net,stride)
    return net

def inception_block_a(net,training,trainable,name):
  with tf.variable_scope(name) as scope:
    with tf.variable_scope('Branch_1') as scope:
      chan_1 = net
      chan_1 = spec_conv2d(chan_1,training,filters = 64,kernel = 1,stride = 1)
      chan_1 = spec_conv2d(chan_1,training,filters = 96,kernel = 3,stride = 1)
      chan_1 = spec_conv2d(chan_1,training,filters = 96,kernel = 3,stride = 1)

    with tf.variable_scope('Branch_2') as scope:
      chan_2 = net
      chan_2 = spec_conv2d(chan_2,training,filters = 64,kernel = 1,stride = 1)
      chan_2 = spec_conv2d(chan_2,training,filters = 96,kernel = 3,stride = 1)

    with tf.variable_scope('Branch_3') as scope:
      chan_3 = net
      chan_3 = spec_conv2d(chan_3,training,filters = 96,kernel = 1,stride = 1)

    with tf.variable_scope('Branch_4') as scope:
      chan_4 = net
      chan_4 = tf.layers.average_pooling2d(chan_4,3,1,padding = 'same')
      chan_4 = spec_conv2d(chan_4,training,filters = 96,kernel = 1,stride = 1)

    net = [chan_1,chan_2,chan_3,chan_4]
    net = ops.delist(net)

  return net

def inception_block_b(net,training,trainable,name):
  with tf.variable_scope(name) as scope:
    with tf.variable_scope('Branch_1') as scope:
      chan_1 = net
      chan_1 = spec_conv2d(chan_1,training,filters = 192,kernel = 1    ,stride = 1)
      chan_1 = spec_conv2d(chan_1,training,filters = 192,kernel = (1,7),stride = 1)
      chan_1 = spec_conv2d(chan_1,training,filters = 224,kernel = (7,1),stride = 1)
      chan_1 = spec_conv2d(chan_1,training,filters = 224,kernel = (1,7),stride = 1)
      chan_1 = spec_conv2d(chan_1,training,filters = 256,kernel = (7,1),stride = 1)

    with tf.variable_scope('Branch_2') as scope:
      chan_2 = net
      chan_2 = spec_conv2d(chan_2,training,filters = 192,kernel = 1    ,stride = 1)
      chan_2 = spec_conv2d(chan_2,training,filters = 224,kernel = (1,7),stride = 1)
      chan_2 = spec_conv2d(chan_2,training,filters = 256,kernel = (7,1),stride = 1)

    with tf.variable_scope('Branch_3') as scope:
      chan_3 = net
      chan_3 = spec_conv2d(chan_3,training,filters = 384,kernel = 1,stride = 1)

    with tf.variable_scope('Branch_4') as scope:
      chan_4 = net
      chan_4 = tf.layers.average_pooling2d(chan_4,3,1,padding = 'same')
      chan_4 = spec_conv2d(chan_4,training,filters = 128,kernel = 1,stride = 1)

    net = [chan_1,chan_2,chan_3,chan_4]
    net = ops.delist(net)

  return net


def inception_block_c(net,training,trainable,name):
  with tf.variable_scope(name) as scope:
    with tf.variable_scope('Branch_1') as scope:
      chan_1   = net
      chan_1   = spec_conv2d(chan_1,training,filters = 384,kernel = 1    ,stride = 1)
      chan_1   = spec_conv2d(chan_1,training,filters = 448,kernel = (1,3),stride = 1)
      chan_1   = spec_conv2d(chan_1,training,filters = 512,kernel = (3,1),stride = 1)
      chan_1_a = spec_conv2d(chan_1,training,filters = 256,kernel = (1,3),stride = 1)
      chan_1_b = spec_conv2d(chan_1,training,filters = 256,kernel = (3,1),stride = 1)
      chan_1   = [chan_1_a,chan_1_b]

    with tf.variable_scope('Branch_2') as scope:
      chan_2   = net
      chan_2   = spec_conv2d(chan_2,training,filters = 384,kernel = 1    ,stride = 1)
      chan_2_a = spec_conv2d(chan_2,training,filters = 256,kernel = (1,3),stride = 1)
      chan_2_b = spec_conv2d(chan_2,training,filters = 256,kernel = (3,1),stride = 1)
      chan_2   = [chan_2_a,chan_2_b]

    with tf.variable_scope('Branch_3') as scope:
      chan_3 = net
      chan_3 = spec_conv2d(chan_3,training,filters = 256,kernel = 1,stride = 1)

    with tf.variable_scope('Branch_4') as scope:
      chan_4 = net
      chan_4 = tf.layers.average_pooling2d(chan_4,3,1,padding = 'same')
      chan_4 = spec_conv2d(chan_4,training,filters = 256,kernel = 1,stride = 1)

    net = [chan_1,chan_2,chan_3,chan_4]
    net = ops.delist(net)

  return net
