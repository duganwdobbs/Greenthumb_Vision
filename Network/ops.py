import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def delist(net):
  if type(net) is list:
    net = tf.concat(net,-1,name = 'cat')
  return net

def lrelu(x):
  return tf.leaky_relu(x)

def conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = lrelu, padding = 'SAME', trainable = True, name = None, reuse = None):
  return tf.layers.conv2d(net,filters,kernel,stride,padding,dilation_rate = dilation_rate, activation = activation,trainable = trainable, name = name, reuse = reuse)

def avg_pool(net, kernel = 3, stride = 1, padding = 'SAME', name = None):
  return tf.layers.average_pooling2d(net,kernel,stride,padding=padding,name=name)

def max_pool(net, kernel = 3, stride = 3, padding = 'SAME', name = None):
  return tf.layers.max_pooling2d(net,kernel,stride,padding=padding,name=name)

def conv2d_trans(net, features, kernel, stride, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconv(net, features = 3, kernel = 3, stride = 2, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def batch_norm(net,training,trainable):
  with tf.variable_scope('batch_norm') as scope:  
    net = tf.layers.batch_normalization(net,training = training, trainable = trainable)
  return net

def deconvxy(net,training, stride = 2,features = None, activation = lrelu,padding = 'SAME', trainable = True, name = 'Deconv_xy'):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    kernel = stride * 2 + stride % 2

    if features is None:
      features = int(net.shape[-1].value / stride)

    netx = deconv(net , features  , kernel = kernel, stride = (stride,1), name = "x", trainable = trainable)
    netx = deconv(netx, features  , kernel = kernel, stride = (1,stride), name = "xy", trainable = trainable)
    nety = deconv(net , features  , kernel = kernel, stride = (1,stride), name = "y", trainable = trainable)
    nety = deconv(nety, features  , kernel = kernel, stride = (stride,1), name = "yx", trainable = trainable)

    net  = tf.concat((netx,nety),-1)

    if FLAGS.batch_norm:
      net = batch_norm(net,training,trainable)

    return net

def dense_block(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = lrelu, padding = 'SAME', trainable = True,
                name = 'Dense_Block', prestride_return = True):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    for n in range(kmap):
      out = conv2d(net,filters=filters,kernel=kernel,stride=1,activation=activation,padding=padding,trainable=trainable,name = '_map_%d'%n)
      net = tf.concat([net,out],-1)

    if FLAGS.batch_norm:
      net = batch_norm(net,training,trainable)

    if stride is not 1:
      prestride = net
      net = max_pool(net,stride,stride)
      if prestride_return:
        return prestride, net
      else:
        return net

    else:
      return net

def atrous_block(net,filters = 8,kernel = 3,dilation = 1,kmap = 2,activation = lrelu,trainable = True,name = 'Atrous_Block'):
  newnet = []
  with tf.variable_scope(name) as scope:
    for x in range(dilation,kmap * dilation,dilation):
      # Reuse and not trainable if beyond the first layer.
      re = True  if x > dilation else None
      tr = False if x > dilation else trainable

      with tf.variable_scope("ATROUS",reuse = tf.AUTO_REUSE) as scope:
        # Total Kernel visual size: Kernel + ((Kernel - 1) * (Dilation - 1))
        # At kernel = 9 with dilation = 2; 9 + 8 * 1, 17 px
        layer = conv2d(net,filters = filters, kernel = kernel, dilation_rate = x,reuse = re,trainable = tr)
        newnet.append(layer)

    net = delist(newnet)
    return net


# Defines a function to output the histogram of trainable variables into TensorBoard
def hist_summ():
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

def cmat(labels_flat,logits_flat):
  with tf.variable_scope("Confusion_Matrix") as scope:
    label_1d  = tf.reshape(labels_flat, (FLAGS.batch_size, FLAGS.imgW * FLAGS.imgH))
    logit_1d = tf.reshape(logits_flat, (FLAGS.batch_size, FLAGS.imgW * FLAGS.imgH))
    cmat_sum = tf.zeros((FLAGS.num_classes,FLAGS.num_classes),tf.int32)
    for i in range(FLAGS.batch_size):
      cmat = tf.confusion_matrix(labels = label_1d[i], predictions = logit_1d[i], num_classes = FLAGS.num_classes)
      cmat_sum = tf.add(cmat,cmat_sum)
    return cmat_sum

def l2loss(loss):
  if FLAGS.l2_loss:
    with tf.variable_scope("L2_Loss") as scope:
      l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])
      l2 = tf.scalar_mul(.0002,l2)
      tf.summary.scalar('L2_Loss',loss)
      loss = tf.add(loss,l2)
      tf.summary.scalar('Total_Loss',loss)
  return loss

# Function to compute Mean Square Error loss
def mse_loss(labels,logits):
  with tf.variable_scope('Mean_Square_Error') as scope:
    loss = tf.losses.mean_squared_error(labels,logits)
    tf.summary.scalar('MSE_Loss',loss)
    loss = l2loss(loss)
    return loss

# A log loss for using single class heat map
def log_loss(labels,logits):
  with tf.variable_scope('Log_Loss') as scope:
    loss = tf.losses.log_loss(labels,logits)
    tf.summary.scalar('Log_Loss',loss)
    loss = l2loss(loss)
    return loss

# Loss function for tape, using cross entropy
def xentropy_loss(labels,logits):
  with tf.variable_scope("XEnt_Loss") as scope:
    labels = tf.cast(labels,tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = logits)
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('XEnt_Loss',loss)
    loss = l2loss(loss)
    return loss

# Absolute accuracy calculation for counting
def accuracy(labels_flat,logits_flat):
  with tf.variable_scope("Accuracy") as scope:
    accuracy = tf.metrics.accuracy(labels = labels_flat, predictions = logits_flat)
    acc,up = accuracy
    tf.summary.scalar('Accuracy',tf.multiply(acc,100))
    return accuracy

def miou(labels,logits):
  with tf.variable_scope("MIOU") as scope:
    miou      = tf.metrics.mean_iou(labels = labels, predictions = logits, num_classes = FLAGS.num_classes)
    _miou,op  = miou
    tf.summary.scalar('MIOU',_miou)
    return miou
