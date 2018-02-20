import tensorflow as tf

def get_shape(source):
  batch, input_height, input_width, channels = source.get_shape().as_list()
  return batch, input_height, input_width, channels

def create_filter(channels,filters,height,width):
  filt = tf.truncated_normal([channels,filters,height,width],mean=0,stddev=.01)
  filt = tf.batch_fft2d(tf.complex(w,0.0*w),name = 'Spectral_Init')
  return filt

def create_bias(features):
  bias = tf.truncated_normal([features],mean=0,stddev=0.01)
  bias = tf.complex(bias,bias)
  return bias

def spec_batch_norm(net,training,trainable):
  if training:
  # Normalize the gradients to the range of 0-360 with a mean of 160 (Real)
  # Calculate and store a moving average of the population mean and variance
  # Normalize the magnitudes to -1 to 1 with a mean of 0             (Imaginary)
  # Calculate and store a moving average of the population mean and variance
  else: # During testing
  # Normalize the gradients using population statistics  (Real)
  # Normalize the magnitudes using population statistics (Imaginary)

  return net

# Spectral conv2d, assumes net comes in as a representation of a FFT
def spec_conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, trainable = True, name = None, reuse = None):
  if net.dtype is not tf.complex64:
    raise Exception("Spectral Conv2d received image not in complex64")
  # Need to define how to reuse
  with tf.variable_scope(name) as scope:
    filt = # Something to create a spectral filter
           # Note, dilation rate must be maintained
           # Note, function to create vars must work with reuse and trainable
    b    = # Something to create a spectral bias
           #
    return net * filt + b

# conv2d_transpose
def spec_conv2d_transpose(net, features, kernel, stride, trainable = True, name = None):
  with tf.variable_scope(name) as scope:
    # Pad image with #stride 0's, then:
    return spec_conv2d(net,features,kernel)

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
