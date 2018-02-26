import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('conv_scope',0,'Incrementer for convolutional scopes')
flags.DEFINE_integer('bn_scope'  ,0,'Incrementer for batch norm scopes')

bn_scope = 0

def init_scope_vars():
  FLAGS.conv_scope = 0
  FLAGS.bn_scope = 0

def delist(net):
  if type(net) is list:
    net = tf.concat(net,-1,name = 'cat')
  return net

def lrelu(x):
  with tf.variable_scope('lrelu') as scope:
    if x.dtype is not tf.complex64:
      return tf.nn.leaky_relu(x)
    else:
      return x

def relu(x):
  with tf.variable_scope('relu') as scope:
    if x.dtype is not tf.complex64:
      return tf.nn.relu(x)
    else:
      return x

def linear(x):
  return x

def conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = relu, padding = 'SAME', trainable = True, name = None, reuse = None):
  net = tf.layers.conv2d(delist(net),filters,kernel,stride,padding,dilation_rate = dilation_rate, activation = activation,trainable = trainable, name = name, reuse = reuse)
  return net

def bn_conv2d(net, training, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = lrelu, use_bias = False, padding = 'SAME', trainable = True, name = None, reuse = None):
  with tf.variable_scope('BN_Conv_%d'%(FLAGS.conv_scope)) as scope:
    FLAGS.conv_scope+=1
    net = conv2d(delist(net), filters, kernel, stride, dilation_rate, activation, padding, trainable, name, reuse)
    net = batch_norm(net,training,trainable,activation)
    return net

def batch_norm(net,training,trainable,activation = relu):
  with tf.variable_scope('Batch_Norm_%d'%(FLAGS.bn_scope)):
    FLAGS.bn_scope = FLAGS.bn_scope + 1
    net = tf.layers.batch_normalization(delist(net),training = training, trainable = trainable)
    if activation is not None:
      net = activation(net)

    return net

def avg_pool(net, kernel = 3, stride = 1, padding = 'SAME', name = None):
  return tf.layers.average_pooling2d(net,kernel,stride,padding=padding,name=name)

def max_pool(net, kernel = 3, stride = 3, padding = 'SAME', name = None):
  return tf.layers.max_pooling2d(net,kernel,stride,padding=padding,name=name)

def conv2d_trans(net, filters, kernel, stride, activation = relu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,filters,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconv(net, filters = 3, kernel = 3, stride = 2, activation = relu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,filters,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconvxy(net,training, stride = 2,filters = None, activation = relu,padding = 'SAME', trainable = True, name = 'Deconv_xy'):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    kernel = stride * 2 + stride % 2

    if filters is None:
      filters = int(net.shape[-1].value / stride)

    netx = deconv(net , filters  , kernel = kernel, stride = (stride,1), name = "x",  trainable = trainable)
    nety = deconv(net , filters  , kernel = kernel, stride = (1,stride), name = "y",  trainable = trainable)

    filters = int(filters / stride)

    netx = deconv(netx, filters  , kernel = kernel, stride = (1,stride), name = "xy", trainable = trainable)
    nety = deconv(nety, filters  , kernel = kernel, stride = (stride,1), name = "yx", trainable = trainable)

    net  = tf.concat((netx,nety),-1)

    if FLAGS.batch_norm:
      net = batch_norm(net,training,trainable)

    return net

def dense_block(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = relu, padding = 'SAME', trainable = True,
                name = 'Dense_Block', prestride_return = True,use_max_pool = True):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    for n in range(kmap):
      out = conv2d(net,filters=filters,kernel=kernel,stride=1,activation=activation,padding=padding,trainable=trainable,name = '_map_%d'%n)
      net = tf.concat([net,out],-1,name = '%d_concat'%n)

    if FLAGS.batch_norm:
      net = batch_norm(net,training,trainable)

    if stride is not 1:
      prestride = net
      if use_max_pool:
        net = max_pool(net,stride,stride)
      else:
        net = avg_pool(net,stride,stride)
      if prestride_return:
        return prestride, net
      else:
        return net

    else:
      return net

def atrous_block(net,training,filters = 8,kernel = 3,dilation = 1,kmap = 2,stride = 1,activation = relu,trainable = True,name = 'Atrous_Block'):
  newnet = []
  with tf.variable_scope(name) as scope:
    for x in range(dilation,kmap * dilation,dilation):
      # Reuse and not trainable if beyond the first layer.
      re = True  if x > dilation else None
      tr = False if x > dilation else trainable

      with tf.variable_scope("ATROUS",reuse = tf.AUTO_REUSE) as scope:
        # Total Kernel visual size: Kernel + ((Kernel - 1) * (Dilation - 1))
        # At kernel = 9 with dilation = 2; 9 + 8 * 1, 17 px
        layer = conv2d(net,filters = filters, kernel = kernel, dilation_rate = x,reuse = re,trainable = tr,padding='SAME')
        newnet.append(layer)

    net = delist(newnet)
    net = bn_conv2d(net,training,filters = filters,kernel = stride,stride = stride,trainable = trainable,name = 'GradientDisrupt',activation = relu)
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

def l2loss(loss,loss_vars = None,l2 = None):
  if l2 is None:
    l2 = FLAGS.l2_loss
  if l2:
    with tf.variable_scope("L2_Loss") as scope:
      loss_vars = tf.trainable_variables() if loss_vars is None else loss_vars
      l2 = tf.add_n([tf.nn.l2_loss(var) for var in loss_vars if 'bias' not in var.name])
      l2 = tf.scalar_mul(.0002,l2)
      tf.summary.scalar('L2_Loss',l2)
      loss = tf.add(loss,l2)
      tf.summary.scalar('Total_Loss',loss)
  return loss

# Function to compute Mean Square Error loss
def mse_loss(labels,logits):
  with tf.variable_scope('Mean_Square_Error') as scope:
    labels = tf.squeeze(labels)
    logits = tf.squeeze(logits)
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
def xentropy_loss(labels,logits,loss_vars = None,l2 = None,name = 'Xent_Loss'):
  with tf.variable_scope(name) as scope:
    labels = tf.cast(labels,tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = logits)
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('XEnt_Loss',loss)
    loss = l2loss(loss,loss_vars,l2)
    return loss

# Absolute accuracy calculation for counting
def accuracy(labels_flat,logits_flat,name = 'Accuracy'):
  with tf.variable_scope(name) as scope:
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

# Relative error calculation for counting
# |Lab - Log| / Possible Classes
def count_rel_err(labels,logits,global_step):
  with tf.variable_scope("rel_err_calc") as scope:
    labels = tf.cast(labels,tf.float32)
    logits = tf.cast(logits,tf.float32)
    sum_acc= tf.Variable(0,dtype = tf.float32,name = 'Sum_Rel_Err')
    g_step = tf.cast(global_step,tf.float32)

    err    = tf.subtract(labels,logits)
    err    = tf.abs(err)
    rel_err= tf.divide(err,FLAGS.num_count_classes)
    rel_err= tf.minimum(1.0,rel_err)
    rel_err   = tf.squeeze(rel_err)
    if(FLAGS.batch_size > 1):
      rel_err   = tf.reduce_mean(rel_err,-1)
    update = tf.assign_add(sum_acc,rel_err)
    value  = sum_acc / g_step

    #tf.summary.scalar('Relative_Error',rel_err)
    tf.summary.scalar('Relative_Error',value)

    return rel_err,update

# Relative Accuracy calculation, I don't like this!
# |Lab - Log| / Lab
def count_rel_acc(labels,logits,global_step):
  with tf.variable_scope("rel_acc_calc") as scope:
    labels = tf.cast(labels,tf.float32)
    logits = tf.cast(logits,tf.float32)
    g_step = tf.cast(global_step,tf.float32)
    sum_acc= tf.Variable(0,dtype = tf.float32,name = 'Sum_Rel_Acc')

    l_min = tf.minimum(labels,logits)
    l_max = tf.maximum(labels,logits)
    rel = tf.divide(tf.abs(tf.subtract(logits,labels)),labels)

    zero = tf.constant(0,dtype=tf.float32,name='zero')
    one  = tf.constant(1,dtype=tf.float32,name='one')
    full_zeros = tf.zeros((FLAGS.batch_size),tf.float32)
    full_zeros = tf.squeeze(full_zeros)
    full_ones  = tf.ones((FLAGS.batch_size),tf.float32)
    full_ones  = tf.squeeze(full_ones)
    full_false = []
    for x in range(FLAGS.batch_size):
      full_false.append(False)
    full_false = tf.convert_to_tensor(full_false)
    full_false = tf.squeeze(full_false)

    # Get where NAN, inf, and logits is zero
    nans  = tf.is_nan(rel)
    infs  = tf.is_inf(rel)
    zeros = tf.equal(logits,zero)

    #If its a zero, get the NaN position, otherwise false.
    z_nan = tf.where(zeros,nans,full_false)
    z_inf = tf.where(zeros,infs,full_false)
    # Set to 1
    rel   = tf.where(z_nan,full_ones,rel)
    rel   = tf.where(z_inf,full_ones,rel)

    # Any leftover NaN or inf is where we counted wrong, so the rel acc is zero.
    nans  = tf.is_nan(rel)
    infs  = tf.is_inf(rel)
    rel   = tf.where(nans,full_zeros,rel)
    rel   = tf.where(infs,full_zeros,rel)

    # Get the minimum of relative acc or 1/ rel acc
    rel   = tf.minimum(rel,tf.divide(one,rel))

    rel = tf.reduce_mean(rel,-1)

    update = tf.assign_add(sum_acc,rel)
    value  = tf.divide(sum_acc,g_step)

    tf.summary.scalar('Relative Accuracy',tf.multiply(value,100))
    return rel,update


def dense_reduction(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = lrelu, trainable = True,name = 'Dense_Block'):
  with tf.variable_scope(name) as scope:
    net = delist(net)
    for n in range(kmap):
      out = bn_conv2d(net, training, filters=filters, kernel=kernel, stride=1,
                        activation=activation, trainable=trainable, name = '_map_%d'%n)
      net = tf.concat([net,out],-1,name = '%d_concat'%n)
    if stride is not 1:
      net = max_pool(net,stride,stride)
    return net

def inception_block_a(net,training,trainable,name):
  with tf.variable_scope(name) as scope:
    with tf.variable_scope('Branch_1') as scope:
      chan_1 = net
      chan_1 = bn_conv2d(chan_1,training,filters = 64,kernel = 1,stride = 1)
      chan_1 = bn_conv2d(chan_1,training,filters = 96,kernel = 3,stride = 1)
      chan_1 = bn_conv2d(chan_1,training,filters = 96,kernel = 3,stride = 1)

    with tf.variable_scope('Branch_2') as scope:
      chan_2 = net
      chan_2 = bn_conv2d(chan_2,training,filters = 64,kernel = 1,stride = 1)
      chan_2 = bn_conv2d(chan_2,training,filters = 96,kernel = 3,stride = 1)

    with tf.variable_scope('Branch_3') as scope:
      chan_3 = net
      chan_3 = bn_conv2d(chan_3,training,filters = 96,kernel = 1,stride = 1)

    with tf.variable_scope('Branch_4') as scope:
      chan_4 = net
      chan_4 = tf.layers.average_pooling2d(chan_4,3,1,padding = 'same')
      chan_4 = bn_conv2d(chan_4,training,filters = 96,kernel = 1,stride = 1)

    net = [chan_1,chan_2,chan_3,chan_4]
    net = delist(net)

  return net

def inception_block_b(net,training,trainable,name):
  with tf.variable_scope(name) as scope:
    with tf.variable_scope('Branch_1') as scope:
      chan_1 = net
      chan_1 = bn_conv2d(chan_1,training,filters = 86,kernel = 1    ,stride = 1)
      chan_1 = bn_conv2d(chan_1,training,filters = 86,kernel = (1,7),stride = 1)
      chan_1 = bn_conv2d(chan_1,training,filters = 112,kernel = (7,1),stride = 1)
      chan_1 = bn_conv2d(chan_1,training,filters = 112,kernel = (1,7),stride = 1)
      chan_1 = bn_conv2d(chan_1,training,filters = 128,kernel = (7,1),stride = 1)

    with tf.variable_scope('Branch_2') as scope:
      chan_2 = net
      chan_2 = bn_conv2d(chan_2,training,filters = 96,kernel = 1    ,stride = 1)
      chan_2 = bn_conv2d(chan_2,training,filters = 112,kernel = (1,7),stride = 1)
      chan_2 = bn_conv2d(chan_2,training,filters = 128,kernel = (7,1),stride = 1)

    with tf.variable_scope('Branch_3') as scope:
      chan_3 = net
      chan_3 = bn_conv2d(chan_3,training,filters = 192,kernel = 1,stride = 1)

    with tf.variable_scope('Branch_4') as scope:
      chan_4 = net
      chan_4 = tf.layers.average_pooling2d(chan_4,3,1,padding = 'same')
      chan_4 = bn_conv2d(chan_4,training,filters = 64,kernel = 1,stride = 1)

    net = [chan_1,chan_2,chan_3,chan_4]
    net = delist(net)

  return net


def inception_block_c(net,training,trainable,name):
  with tf.variable_scope(name) as scope:
    with tf.variable_scope('Branch_1') as scope:
      chan_1   = net
      chan_1   = bn_conv2d(chan_1,training,filters = 384,kernel = 1    ,stride = 1)
      chan_1   = bn_conv2d(chan_1,training,filters = 448,kernel = (1,3),stride = 1)
      chan_1   = bn_conv2d(chan_1,training,filters = 512,kernel = (3,1),stride = 1)
      chan_1_a = bn_conv2d(chan_1,training,filters = 256,kernel = (1,3),stride = 1)
      chan_1_b = bn_conv2d(chan_1,training,filters = 256,kernel = (3,1),stride = 1)
      chan_1   = [chan_1_a,chan_1_b]
      chan_1   = delist(chan_1)

    with tf.variable_scope('Branch_2') as scope:
      chan_2   = net
      chan_2   = bn_conv2d(chan_2,training,filters = 384,kernel = 1    ,stride = 1)
      chan_2_a = bn_conv2d(chan_2,training,filters = 256,kernel = (1,3),stride = 1)
      chan_2_b = bn_conv2d(chan_2,training,filters = 256,kernel = (3,1),stride = 1)
      chan_2   = [chan_2_a,chan_2_b]
      chan_2   = delist(chan_2)

    with tf.variable_scope('Branch_3') as scope:
      chan_3 = net
      chan_3 = bn_conv2d(chan_3,training,filters = 256,kernel = 1,stride = 1)

    with tf.variable_scope('Branch_4') as scope:
      chan_4 = net
      chan_4 = tf.layers.average_pooling2d(chan_4,3,1,padding = 'same')
      chan_4 = bn_conv2d(chan_4,training,filters = 256,kernel = 1,stride = 1)

    net = [chan_1,chan_2,chan_3,chan_4]
    net = delist(net)

  return net
