import os
import time
import platform

import numpy         as     np
import tensorflow    as     tf

from multiprocessing import Process
from inspector       import inspect
from util            import factors
from util            import ImageToPatch
from util            import PatchToImage
from tfrecord        import inputs
from tfrecord        import sizes
from time            import sleep


imgW,imgH,save_stride = sizes()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'D:/Cows/','Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/BinaLab-Semantic-Segmentation/data/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss' ,False,'If we use l2 regularization')
flags.DEFINE_boolean('batch_norm' ,True ,'If we use batch normalization')
flags.DEFINE_boolean('lr_decay',True,'If we use Learning Rate Decay')
flags.DEFINE_boolean('advanced_logging' ,False,'If we log metadata and histograms')
flags.DEFINE_integer('num_epochs'       , None                ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size'       , 1                 ,'Batch size for training.')
flags.DEFINE_integer('train_steps'      ,10000                ,'Number of steps for training on counting')
flags.DEFINE_integer('num_classes'      ,1                    ,'# Classes')
flags.DEFINE_string ('run_dir'    , FLAGS.base_dir + '/network_log/','Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  ,FLAGS.base_dir  + '/'            ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  ,'cows.ckt'                  ,'Checkpoint name')
flags.DEFINE_string ('ckpt_i_name','cows-interrupt.ckpt'                  ,'Checkpoint name')

def launchTensorBoard(directory):
  sleep(30)
  os.system(directory)

def lrelu(x):
  return tf.maximum(0.1 * x, x)

def delist(net):
  if type(net) is list:
    net = tf.concat(net,-1,name = 'cat')
  return net

def training(global_step,loss,train_vars,learning_rate = .001):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.variable_scope("Optimizer") as scope:
      if FLAGS.lr_decay:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   250, 0.85, staircase=True)
      optomizer = tf.train.RMSPropOptimizer(learning_rate,momentum = .8, epsilon = 1e-5)
      train     = optomizer.minimize(loss,var_list = train_vars,global_step=global_step)
  return train

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

    if FLAGS.batch_norm:
      net = batch_norm(net,training,trainable)

    kernel = stride * 2 + stride % 2

    if features is None:
      features = int(net.shape[-1].value / stride)

    netx = deconv(net , features  , kernel = kernel, stride = (stride,1), name = "x", trainable = trainable)
    netx = deconv(netx, features  , kernel = kernel, stride = (1,stride), name = "xy", trainable = trainable)
    nety = deconv(net , features  , kernel = kernel, stride = (1,stride), name = "y", trainable = trainable)
    nety = deconv(nety, features  , kernel = kernel, stride = (stride,1), name = "yx", trainable = trainable)

    net  = tf.concat((netx,nety),-1)
    net  = conv2d(net,features,kernel = 1,name = 'compresor',trainable = trainable)
    return net

def dense_block(net,training,filters = 2,kernel = 3,kmap = 5,stride = 1,activation = lrelu, padding = 'SAME',trainable = True, name = 'Dense_Block'):
  with tf.variable_scope(name) as scope:

    net = delist(net)

    if FLAGS.batch_norm:
      net = batch_norm(net,training,trainable)

    for n in range(kmap):
      out = conv2d(net,filters=filters,kernel=kernel,stride=1,activation=activation,padding=padding,trainable=trainable,name = '_map_%d'%n)
      net = tf.concat([net,out],-1)

    if stride is not 1:
      prestride = net
      net = max_pool(net,stride,stride)
      return prestride, net

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

# The segmentation network.
def inference(images,training,name,trainable = True):
  with tf.variable_scope(name) as scope:
    net = images
    net = tf.cast(net,tf.float32)
    net = batch_norm(net,training,trainable)

    net = atrous_block(net,filters = 2, kernel = 4, dilation = 2, kmap = 3, trainable = trainable)

    strides = factors(imgH,imgW)
    prestride = []

    for x in range(len(strides)):
      pre, net = dense_block(net,training = training, filters = 4,kernel = 3,  kmap = 3  ,stride = strides[x],trainable = trainable,
                             name = 'DenseBlock%d'%x)
      prestride.append(pre)

    for x in range(len(strides)):
      net = deconvxy(net,training,stride = strides[-(x+1)], trainable = trainable,
                     name = 'DeconvXY%d'%x)
      net = [net,prestride[-(x+1)]]

    net = delist(net)

    logits = conv2d(net,FLAGS.num_classes,5,stride = 1,name = 'finale',trainable = trainable)
    return logits

# Defines a function to output the histogram of trainable variables into TensorBoard
def hist_summ():
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

# Absolute accuracy calculation for counting
def accuracy(labels_flat,logits_flat):
  with tf.variable_scope("Accuracy") as scope:
    accuracy = tf.metrics.accuracy(labels = labels_flat, predictions = logits_flat)
    acc,up = accuracy
    tf.summary.scalar('Accuracy',tf.multiply(acc,100))
    return accuracy

def cmat(labels_flat,logits_flat):
  with tf.variable_scope("Confusion_Matrix") as scope:
    label_1d  = tf.reshape(labels_flat, (FLAGS.batch_size, imgW * imgH))
    logit_1d = tf.reshape(logits_flat, (FLAGS.batch_size, imgW * imgH))
    cmat_sum = tf.zeros((FLAGS.num_classes,FLAGS.num_classes),tf.int32)
    for i in range(FLAGS.batch_size):
      cmat = tf.confusion_matrix(labels = label_1d[i], predictions = logit_1d[i], num_classes = FLAGS.num_classes)
      cmat_sum = tf.add(cmat,cmat_sum)
    return cmat_sum

def l2loss():
  with tf.variable_scope("L2_Loss") as scope:
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])
    l2 = tf.scalar_mul(.0002,l2)
    tf.summary.scalar('L2_Loss',loss)
    return l2

# Function to compute Mean Square Error loss
def mse_loss(labels,logits):
  with tf.variable_scope('Mean_Square_Error') as scope:
    loss = tf.losses.mean_squared_error(labels,logits)
    tf.summary.scalar('MSE_Loss',loss)

    if FLAGS.l2_loss:
      loss = tf.add(loss,l2loss())
    tf.summary.scalar('Total_Loss',loss)
    return loss


# Loss function for tape, using cross entropy
def xentropy_loss(labels,logits):
  with tf.variable_scope("XEnt_Loss") as scope:
    logits = tf.reshape(logits,[FLAGS.batch_size*imgH*imgW,FLAGS.num_classes])
    labels = tf.reshape(tf.argmax(labels,-1),[FLAGS.batch_size*imgH*imgW])
    labels = tf.cast(labels,tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = logits)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('XEnt_Loss',loss)

    if FLAGS.l2_loss:
      loss = tf.add(loss,l2loss())
    tf.summary.scalar('Total_Loss',loss)
    return loss

def per_class_acc(labels,logits):
  with tf.variable_scope("PerClassAcc"):
    pcacc = tf.metrics.mean_per_class_accuracy(labels = labels, predictions = logits, num_classes = FLAGS.num_classes)
    p_acc, op = p_acc


def miou(labels,logits):
  with tf.variable_scope("MIOU") as scope:
    miou      = tf.metrics.mean_iou(labels = labels, predictions = logits, num_classes = FLAGS.num_classes)
    _miou,op  = miou
    tf.summary.scalar('MIOU',_miou)
    return miou

# Image saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def save_imgs(images):
  with tf.variable_scope("Img_Save") as scope:
    # Setting up the save operations
    save_imgs      = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(images[i,:,:,:],3)
      save_imgs.append(save_img)

    tf.summary.image('Image',images[:,::save_stride,::save_stride,:])
    save_imgs_grouped = tf.tuple(save_imgs)
    return save_imgs_grouped

# Tape GT saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def save_labs(labels):
  with tf.variable_scope("GT_Save") as scope:
    # labels = tf.reshape(labels,[FLAGS.batch_size,imgH,imgW])
    intensity = tf.convert_to_tensor(255,dtype = tf.float32,name = 'intensity')
    labels = tf.scalar_mul(intensity,labels)
    labels = tf.cast(labels,tf.uint8)
    save_gts       = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(labels[i,:,:],1)
      save_gts.append(save_img)

    tf.summary.image('Ground_Truth',labels[:,::save_stride,::save_stride,:])
    save_gts_grouped  = tf.tuple(save_gts)
    return save_gts_grouped

# Tape Logit saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def save_logs(logits):
  with tf.variable_scope("Pred_Save") as scope:
    # logits = tf.reshape(logits,[FLAGS.batch_size,imgH,imgW])
    intensity = tf.convert_to_tensor(255,dtype = tf.float32,name = 'intensity')
    logits = tf.scalar_mul(intensity,logits)
    logits = tf.cast(logits,tf.uint8)
    save_logs      = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(logits[i,:,:],1)
      save_logs.append(save_img)

    tf.summary.image('Prediction',logits[:,::save_stride,::save_stride,:])
    save_logs_grouped = tf.tuple(save_logs)
    return save_logs_grouped

# Runs the tape training.
def train(train_run = True, restore = False):
  with tf.Graph().as_default():
      config         = tf.ConfigProto(allow_soft_placement = True)
      sess           = tf.Session(config = config)
      # Setting up a new session
      global_step    = tf.Variable(1,name='global_step',trainable=False)

      if not train_run:
        FLAGS.batch_size = 1
        FLAGS.num_epochs = 1

      # Build the network from images, inference, loss, and backpropogation.
      with tf.variable_scope("Net_Inputs") as scope:
        images, labels, count = inputs(train=train_run,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs,num_classes = FLAGS.num_classes)

      logits         = inference(images,training = train_run,name = 'FCNN',trainable = True)
      # Calculating all fo the metrics, thins we judge a network by
      with tf.variable_scope("Metrics") as scope:
        # Showing var histograms and distrobutions during testing is worthless.
        if FLAGS.advanced_logging and train_run:
          hist_summ()
        loss           = mse_loss(labels,logits)

        metrics        = loss

      train_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'FCNN') if var in tf.trainable_variables()]
      write_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'batch_norm' in var.name]
      if train_run:
        train = training(global_step,loss,train_vars)
      else:
        train = tf.assign_add(global_step,1,name = 'Global_Step')

      # Save operations
      save_imgs_grouped = save_imgs(images)
      save_logs_grouped = save_logs(logits)
      save_labs_grouped  = save_labs(labels)

      # Initialize all variables in network
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())

      # Setting up Tensorboard
      summaries      = tf.summary.merge_all()
      if FLAGS.advanced_logging:
        run_options    = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata   = tf.RunMetadata()
      else:
        run_options    = None
        run_metadata   = None

      timestr        = time.strftime("%d_%b_%Y_%H_%M_TRAIN",time.localtime()) if train_run else time.strftime("%d_%b_%Y_%H_%M_TEST",time.localtime())
      # Location to log to.
      filestr        = FLAGS.run_dir + "tensorlogs/" + timestr + "/"
      print("Running from: ",end='')
      print(filestr)
      # Location and name to save checkpoint.
      savestr        = FLAGS.run_dir + FLAGS.ckpt_name
      # Location and name to inspect checkpoint for logging.
      logstr         = filestr + 'model.ckpt'
      # Setting up summary writer.
      writer         = tf.summary.FileWriter(filestr,sess.graph)

      # Setting up the checkpoint saver and training coordinator for the network
      saver          = tf.train.Saver(train_vars + write_vars)
      if(restore or not train_run):
        # inspect(tf.train.latest_checkpoint(FLAGS.run_dir))
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.run_dir))
        if not train_run:
          # Save the loaded testing checkpoint in the log folder.
          saver.save(sess,logstr)

      # Starts the input generator
      coord          = tf.train.Coordinator()
      threads        = tf.train.start_queue_runners(sess = sess, coord = coord)

      try:
        ops = [train,summaries,metrics] if train_run else [train,summaries,metrics,save_imgs_grouped,save_logs_grouped,save_labs_grouped]

        step = tf.train.global_step(sess,global_step)
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network and write summaries
          if train_run:
            _,_summ_result,_metrics                   = sess.run(ops, options = run_options, run_metadata = run_metadata)
          else:
            _,_summ_result,_metrics,_imgs,_labs,_logs = sess.run(ops, options = run_options, run_metadata = run_metadata)

          # Write summaries
          if FLAGS.advanced_logging:
            writer.add_run_metadata(run_metadata,'step%d'%step)
          writer.add_summary(_summ_result,step)

          #Write the cmat to a file at each step, write images if testing.
          if not train_run:
            for x in range(FLAGS.batch_size):
              with open(filestr + '%d_%d_img.png'%(step,x),'wb+') as f:
                f.write(_imgs[x])
              with open(filestr + '%d_%d_log.png'%(step,x),'wb+') as f:
                f.write(_labs[x])
              with open(filestr + '%d_%d_lab.png'%(step,x),'wb+') as f:
                f.write(_logs[x])

          if step%100 == 0 and train_run:
            # Save some checkpoints
            saver.save(sess,savestr,global_step = step)
            saver.save(sess,logstr,global_step = step)
      except KeyboardInterrupt:
        if train_run:
          saver.save(sess,savestr,global_step = step)
          saver.save(sess,logstr,global_step = step)
      except tf.errors.OutOfRangeError:
        print("Sumthin messed up man.")
      finally:
        if train_run:
          saver.save(sess,savestr)
          saver.save(sess,logstr)
        coord.request_stop()
      coord.join(threads)
      sess.close()

def main(_):
  # TensorBoard = Process(target = launchTensorBoard, args = ('tensorboard --logdir=' + FLAGS.run_dir + "tensorlogs/",))
  #TensorBoard.start()
  train(restore = False)
  train(train_run = False, restore = False)
  #TensorBoard.terminate()
  #TensorBoard.join()

if __name__ == '__main__':
  tf.app.run()
