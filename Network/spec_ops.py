import tensorflow as tf

def activation(net):
  return tf.nn.leaky_relu(net)

def get_shape(source):
  batch, input_height, input_width, channels = source.get_shape().as_list()
  return batch, input_height, input_width, channels

def create_filter(filters,net):
  batch, height, width, channels = source.get_shape().as_list()
  r = tf.truncated_normal([channels,filters,height,width],mean=0,stddev=.01)
  i = tf.truncated_normal([channels,filters,height,width],mean=0,stddev=.01)
  filt = tf.conplex(r,i)
  return filt

def create_bias(features):
  r = tf.truncated_normal([features],mean=0,stddev=.01)
  i = tf.truncated_normal([features],mean=0,stddev=.01)
  bias = tf.complex(r,i)
  return bias

# Spectral conv2d, assumes net comes in as a representation of a FFT
def spec_conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, trainable = True, name = None, reuse = None):
  if net.dtype is not tf.complex64:
    raise Exception("Spectral Conv2d received image not in complex64")
  # Need to define how to reuse
  with tf.variable_scope(name) as scope:
    filt = create_filter(net,)
           # TODO: Note, dilation rate must be maintained
           # TODO: Note, function to create vars must work with reuse and trainable
    b    = create_bias(filtrs)
    net  = net * filt + b
    net  = tf.layers.batch_norm
    return

# conv2d_transpose
def spec_conv2d_transpose(net, features, kernel, stride, trainable = True, name = None):
  with tf.variable_scope(name) as scope:
    # Pad image with #stride 0's, then:
    return spec_conv2d(net,features,kernel)

# dense

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
