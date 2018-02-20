import tensorflow as tf
from tfrecord import inputs
from multiprocessing import Process
from FCount import getFiveClassArr
import numpy as np
from time import sleep
from FCount import getFiveClassArr
import h5py

import time
import os
import platform

imgW        = int(1920/4)
imgH        = int(1080/4)
save_stride = 5

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'E:/BinaLab-Semantic-Segmentation/data/','Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/BinaLab-Semantic-Segmentation/data/','Base os specific DIR')

flags.DEFINE_integer('num_epochs'       , None                ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size'       , 24                  ,'Batch size for training.')
flags.DEFINE_integer('train_steps'      ,12000                ,'Number of steps for training on counting')
flags.DEFINE_integer('num_classes'      ,5                    ,'# Classes')
flags.DEFINE_string ('run_dir'  , FLAGS.base_dir + '/network_log/','Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir',FLAGS.base_dir  + '/'            ,'Location of the tfrecord files.')

def launchTensorBoard(directory):
  sleep(30)
  os.system(directory)

def lrelu(x):
  return tf.maximum(0.1 * x, x)

def conv2d(net, filters, kernel = 3, stride = 1, activation = lrelu, padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d(net,filters,kernel,stride,padding,activation = activation,trainable = trainable, name = name)

def dense_conv(net, filters, kernel = 3, stride = 1, activation = lrelu, padding = 'SAME', trainable = True, name = None):
  with tf.variable_scope(name) as scope:
    return tf.concat([net,conv2d(net,filters,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)],3)

def avg_pool(net, kernel = 3, stride = 1, padding = 'SAME', name = None):
    return tf.layers.average_pooling2d(net,kernel,stride,padding=padding,name=name)

def max_pool(net, filters, kernel = 3, stride = 1, padding = 'SAME', name = None):
    return tf.layers.max_pooling2d(net,kernel,stride,padding=padding,name=name)

def conv2d_trans(net, features, kernel, stride, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconv(net, features = 3, kernel = 3, stride = 2, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name)

def deconvxy(net,train_run,features = None, stride = 2, activation = lrelu,padding = 'SAME', trainable = True, name = None):
  with tf.variable_scope(name) as scope:
    tf.layers.batch_normalization(net,training = train_run)
    kernel = stride + 1
    if features is None:
      features = int(net.shape[-1].value / stride)
    netx = deconv(net , features  , kernel = kernel, stride = (stride,1), name = name +  "deconv_x1", trainable = trainable)
    netx = deconv(netx, features  , kernel = kernel, stride = (1,stride), name = name +  "deconv_xy", trainable = trainable)
    nety = deconv(net , features  , kernel = kernel, stride = (1,stride), name = name +  "deconv_y1", trainable = trainable)
    nety = deconv(nety, features  , kernel = kernel, stride = (stride,1), name = name +  "deconv_yx", trainable = trainable)
    net  = tf.concat((netx,nety),-1,name = name + 'concat')
    net  = conv2d(net,features,kernel = 1,name = name + 'compresor',trainable = trainable)
    return net

def dense_block(net,train_run,filters = 2,kernel = 3,kmap = 5,activation = lrelu, padding = 'SAME',trainable = True, name = None):
  with tf.variable_scope(name) as scope:
    tf.layers.batch_normalization(net,training = train_run)
    for n in range(kmap):
      net = dense_conv(net,filters,kernel,1,activation,padding,trainable,name = name + '_block_%d'%n)
    tf.layers.dropout(net,.2)
    return net

# The segmentation network.
def tape_inference(images,train_run,name,trainable = True):
  with tf.variable_scope("CNN") as scope:
    net = images
    with tf.device('/gpu:1'):
      net  = dense_block(net,train_run,6,kernel = 10,kmap = 3,trainable = trainable,name = 'denseblock_1')
      netFULL = net
      net = avg_pool(net,5,5)
      net  = dense_block(net,train_run,8,kernel = 6,kmap = 6,trainable = trainable,name = 'denseblock_2')
      netFIVE = net

      net = avg_pool(net,3,3)
      net  = dense_block(net,train_run,10,kernel = 4,kmap = 12,trainable = trainable,name = 'denseblock_3')
      netTHREE = net

      net = avg_pool(net,2,2)
      net  = dense_block(net,train_run,12,kernel = 3,kmap = 16,trainable = trainable,name = 'denseblock_4')
      net = deconvxy(net,train_run,stride = 2,trainable = trainable,name = 'deconv2')

      net = tf.concat([net,netTHREE],-1)
      net = deconvxy(net,train_run,24,stride = 3,trainable = trainable,name = 'deconv3')

      net = tf.concat([net,netFIVE],-1)
      net = deconvxy(net,train_run,6,stride = 5,trainable = trainable,name = 'deconv5')

      net = tf.concat([net,netFULL],-1)
      logits = conv2d(net,FLAGS.num_classes,3,name = 'finale',trainable = trainable)
      return logits

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

# Loss function for tape, using cross entropy
def cross_entropy_loss(labels,logits):
  with tf.variable_scope("Loss") as scope:
    logits = tf.reshape(logits,[FLAGS.batch_size*imgH*imgW,FLAGS.num_classes])
    labels = tf.reshape(tf.argmax(labels,-1),[FLAGS.batch_size*imgH*imgW])
    labels = tf.cast(labels,tf.int32)
    freq   = getFiveClassArr()
    freq   = tf.constant(freq,dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,logits = logits)
    # Weighting the loss.
    loss = tf.multiply(freq,loss)
    loss = tf.reduce_mean(loss)

    with tf.variable_scope("L2_Reg") as scope:
      l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])
      l2 = tf.scalar_mul(.0002,l2)

    loss = tf.add(loss,l2)
    tf.summary.scalar('Loss',loss)
    return loss

def precision(labels,logits):
  with tf.variable_scope("Precision") as scope:
    precision = tf.metrics.precision(labels = labels, predictions = logits)
    prec,up   = precision
    tf.summary.scalar('Precision',prec)
    return precision

def recall(labels,logits):
  with tf.variable_scope("Recall") as scope:
    recall   = tf.metrics.recall(labels = labels, predictions = logits)
    rec,up  = recall
    tf.summary.scalar('Recall',rec)
    return recall

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
    intensity      = tf.convert_to_tensor(255,dtype=tf.float32,name='intensity')
    input_images   = tf.add(images,.5)
    input_images   = tf.scalar_mul(intensity,input_images)
    input_images   = tf.cast(input_images,tf.uint8)
    save_imgs      = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(input_images[i,:,:,:],3)
      save_imgs.append(save_img)

    tf.summary.image('Image',input_images[:,::save_stride,::save_stride,:])
    save_imgs_grouped = tf.tuple(save_imgs)
    return save_imgs_grouped

# Tape GT saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def save_labs(labels):
  with tf.variable_scope("GT_Save") as scope:
    intensity_u    = tf.convert_to_tensor(255 / FLAGS.num_classes,dtype=tf.float32,name='intensity_uint8')
    input_labels   = tf.reshape(labels,[FLAGS.batch_size,imgH,imgW,1])
    input_labels    = tf.cast(input_labels,dtype=tf.float32)
    input_labels   = tf.scalar_mul(intensity_u,input_labels)
    input_labels   = tf.cast(input_labels,tf.uint8)
    save_gts       = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(input_labels[i,:,:],1)
      save_gts.append(save_img)

    tf.summary.image('Ground_Truth',input_labels[:,::save_stride,::save_stride,:])
    save_gts_grouped  = tf.tuple(save_gts)
    return save_gts_grouped

# Tape Logit saving function, when the op is called preforms the summary saving,
# but also returns encoded images for saving if needed.
def save_logs(logits):
  with tf.variable_scope("Pred_Save") as scope:
    intensity_u    = tf.convert_to_tensor(255 / FLAGS.num_classes,dtype=tf.float32,name='intensity_uint8')
    save_logits    = tf.reshape(logits,[FLAGS.batch_size,imgH,imgW,1])
    save_logits    = tf.cast(save_logits,dtype=tf.float32)
    save_logits    = tf.scalar_mul(intensity_u,save_logits)
    save_logits    = tf.cast(save_logits,tf.uint8)
    save_logs      = []
    for i in range(FLAGS.batch_size):
      save_img = tf.image.encode_png(save_logits[i,:,:],1)
      save_logs.append(save_img)

    tf.summary.image('Prediction',save_logits[:,::save_stride,::save_stride,:])
    save_logs_grouped = tf.tuple(save_logs)
    return save_logs_grouped

# Runs the tape training.
def train(train_run = True, restore = False):
  with tf.Graph().as_default():
      config         = tf.ConfigProto(allow_soft_placement = True)
      sess           = tf.Session(config = config)
      # Setting up a new session
      global_step    = tf.Variable(1,name='global_step',trainable=False)

      # Build the network from images, inference, loss, and backpropogation.
      with tf.variable_scope("Net_Inputs") as scope:
        images, labels, labels_flat = inputs(train=train_run,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs,num_classes = FLAGS.num_classes)
      logits         = tape_inference(images,train_run,'FCNN',train_run)
      # Calculating all fo the metrics, thins we judge a network by
      with tf.variable_scope("Metrics") as scope:
        logits_flat    = tf.nn.softmax(logits,-1)
        logits_flat    = tf.argmax(logits_flat,-1)
        loss           = cross_entropy_loss(labels,logits)
        net_miou       =          miou(labels_flat,logits_flat)
        net_acc        =      accuracy(labels_flat,logits_flat)
        net_cmat       =          cmat(labels_flat,logits_flat)

        metrics        = net_miou,net_acc

      if train_run:
          with tf.variable_scope("Adam") as scope:
            train        = tf.train.RMSPropOptimizer(.001,momentum = .65,epsilon=1e-3).minimize(loss,global_step=global_step)

      # Save operations
      with tf.variable_scope("ImageSaving") as scope:
        save_imgs_grouped = save_imgs(images)
        save_logs_grouped = save_logs(logits_flat)
        save_gts_grouped  = save_labs(labels_flat)

      # Initialize all variables in network
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())

      # Setting up Tensorboard
      summaries      = tf.summary.merge_all()
      timestr        = time.strftime("TRAIN_%d_%b_%Y_%H_%M_%S",time.localtime())
      filestr        = FLAGS.run_dir + "tensorlogs/" + timestr + "/"
      writer         = tf.summary.FileWriter(filestr,sess.graph)

      # Setting up the checkpoint saver and training coordinator for the network

      if(restore):
        r_vars = [var for var in tf.trainable_variables()]
        saver = tf.train.Saver(r_vars)
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.run_dir))
      saver          = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

      # Starts the input generator
      coord          = tf.train.Coordinator()
      threads        = tf.train.start_queue_runners(sess = sess, coord = coord)

      try:
        # Set the variable in a higher scope

        ops = [train,summaries,net_cmat,metrics]
        step = tf.train.global_step(sess,global_step)
        _cmat = np.zeros((FLAGS.num_classes,FLAGS.num_classes))
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network
          _,_summ_result,_net_cmat,_metrics = sess.run(ops)
          # Write summaries
          writer.add_summary(_summ_result,step)
          # Build the confusion matrix
          _cmat = _cmat + _net_cmat
          #Write the cmat to a file at each step.
          with open(FLAGS.run_dir + "cmat.dat", 'w') as f:
            f.write(str(step))
            f.write(" ")
            f.write(np.array2string(_cmat))

          if(step%100 == 0):
            # Save some checkpoints
            savestr = FLAGS.run_dir + "Train5class.ckpt"
            saver.save(sess,savestr,global_step = step)
      except KeyboardInterrupt:
        savestr = FLAGS.run_dir + "Train5class_interrupt.ckpt"
        saver.save(sess,savestr,global_step = step)
      except tf.errors.OutOfRangeError:
        print("Done.")
      finally:
        savestr = FLAGS.run_dir + "Train5class.ckpt"
        saver.save(sess,savestr)
        savestr = FLAGS.run_dir + "tensorlogs/" + timestr + "/" + "Train5class.ckpt"
        saver.save(sess,savestr)
        coord.request_stop()
      coord.join(threads)
      sess.close()

# Runs the tape training.
def test(train_run = False):
  with tf.Graph().as_default():
      config         = tf.ConfigProto(allow_soft_placement = True)
      sess           = tf.Session(config = config)
      # Setting up a new session
      global_step    = tf.Variable(1,name='global_step',trainable=False)

      # Build the network from images, inference, loss, and backpropogation.
      with tf.variable_scope("Net_Inputs") as scope:
        images, labels, labels_flat = inputs(train=train_run,batch_size=FLAGS.batch_size,num_epochs=1,num_classes = FLAGS.num_classes)
      logits         = tape_inference(images,train_run,'tape',True)
      # Calculating all fo the metrics, thins we judge a network by
      with tf.variable_scope("Metrics") as scope:
        logits_flat    = tf.nn.softmax(logits,-1)
        logits_flat    = tf.argmax(logits_flat,-1)
        loss           = cross_entropy_loss(labels,logits)
        net_prec       =     precision(labels_flat,logits_flat)
        net_rec        =        recall(labels_flat,logits_flat)
        net_miou       =          miou(labels_flat,logits_flat)
        net_acc        =      accuracy(labels_flat,logits_flat)
        net_cmat       =          cmat(labels_flat,logits_flat)

        metrics        = net_prec,net_rec,net_miou,net_acc

      if train_run:
        with tf.variable_scope("Adam") as scope:
          train        = tf.train.AdamOptimizer(.001,epsilon=1e-3).minimize(loss,global_step=global_step)

      else:
        train = tf.assign_add(global_step,1,name = 'Global_Step')

      # Save operations
      with tf.variable_scope("ImageSaving") as scope:
        save_imgs_grouped = save_imgs(images)
        save_logs_grouped = save_logs(logits_flat)
        save_labs_grouped  = save_labs(labels_flat)

      # Initialize all variables in network
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())

      # Setting up Tensorboard
      summaries      = tf.summary.merge_all()
      timestr        = time.strftime("%d_%b_%Y_%H_%M_%S_TEST",time.localtime())
      filestr        = FLAGS.run_dir + "tensorlogs/" + timestr + "/"
      writer         = tf.summary.FileWriter(filestr,sess.graph)

      # Setting up the checkpoint saver and training coordinator for the network

      saver          = tf.train.Saver(tf.trainable_variables())
      saver.restore(sess,FLAGS.run_dir + "Train5class.ckpt")

      savestr = FLAGS.run_dir + "tensorlogs/" + timestr + "/Train5class.ckpt"
      saver.save(sess,savestr)

      # Starts the input generator
      coord          = tf.train.Coordinator()
      threads        = tf.train.start_queue_runners(sess = sess, coord = coord)

      try:
        # Set the variable in a higher scope

        ops = [train,summaries,net_cmat,metrics,save_imgs_grouped,save_logs_grouped,save_labs_grouped]
        step = tf.train.global_step(sess,global_step)
        _cmat = np.zeros((FLAGS.num_classes,FLAGS.num_classes))
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network
          _,_summ_result,_net_cmat,_metrics,_imgs,_labs,_logs = sess.run(ops)
          # Write summaries
          writer.add_summary(_summ_result,step)
          # Build the confusion matrix
          _cmat = _cmat + _net_cmat
          #Write the cmat to a file at each step.
          for x in range(FLAGS.batch_size):
            with open(FLAGS.run_dir + "tensorlogs/" + timestr + '/%d_%d_img.png'%(step,x),'wb+') as f:
              f.write(_imgs[x])
            with open(FLAGS.run_dir + "tensorlogs/" + timestr + '/%d_%d_lab.png'%(step,x),'wb+') as f:
              f.write(_labs[x])
            with open(FLAGS.run_dir + "tensorlogs/" + timestr + '/%d_%d_log.png'%(step,x),'wb+') as f:
              f.write(_logs[x])
          with open(FLAGS.run_dir + "tensorlogs/" + timestr + "/cmat_test.dat", 'w') as f:
            f.write(str(step))
            f.write(" ")
            f.write(np.array2string(_cmat))
      except KeyboardInterrupt:
        pass
      except tf.errors.OutOfRangeError:
        print("Done.")
      finally:
        coord.request_stop()
      coord.join(threads)
      sess.close()



def main(_):
  TensorBoard = Process(target = launchTensorBoard, args = ('tensorboard --logdir=' + FLAGS.run_dir + "tensorlogs/",))
  #TensorBoard.start()
  train(restore = True)
  test()
  #TensorBoard.terminate()
  #TensorBoard.join()

if __name__ == '__main__':
  tf.app.run()
