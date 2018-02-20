import os
import time
import platform

#Custom Imports
import ops
import util

# Aliased Imports
import numpy         as     np
import tensorflow    as     tf

# Selective Imports
from multiprocessing import Process
from inspector       import inspect
from tfrecord        import inputs
from tfrecord        import sizes
from time            import sleep

from util            import factors
from util            import Image_To_Patch
from util            import Patch_To_Image
from util            import ImSizeToPatSize
from util            import disc_label_gen

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'D:/Cows/','Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/data0/ddobbs/Cows/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss'    , False                              ,'If we use l2 regularization')
flags.DEFINE_boolean('batch_norm' , True                               ,'If we use batch normalization')
flags.DEFINE_boolean('lr_decay'   , False                              ,'If we use Learning Rate Decay')
flags.DEFINE_boolean('adv_logging', False                              ,'If we log metadata and histograms')
flags.DEFINE_integer('num_epochs' , None                               ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size' , 1                                  ,'Batch size for training.')
flags.DEFINE_integer('patch_size' , 100                                ,'Patch Size')
flags.DEFINE_integer('imgW'       , 4000                               ,'Image Width')
flags.DEFINE_integer('imgH'       , 3000                               ,'Image Height')
flags.DEFINE_integer('train_steps', 10000                              ,'Number of steps for training on counting')
flags.DEFINE_integer('num_classes', 1                                  ,'# Classes')
flags.DEFINE_integer('save_stride', 10                                 ,'Amount of striding to take when saving images')
flags.DEFINE_string ('run_dir'    , FLAGS.base_dir  + '/network_log/'  ,'Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  , FLAGS.base_dir  + '/'              ,'Location of the tfrecord files.')
flags.DEFINE_string ('net_name'   , 'CowNet'                           ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  , 'CowNet' + '.ckt'            ,'Checkpoint name')
flags.DEFINE_string ('ckpt_i_name', 'CowNet' + '-interrupt.ckpt','Interrupt Checkpoint name')

def launchTensorBoard(directory):
  sleep(30)
  os.system(directory)

def optomize(global_step,loss,train_vars,learning_rate = .001):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.variable_scope("Optimizer") as scope:
      if FLAGS.lr_decay:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   250, 0.85, staircase=True)
      optomizer = tf.train.RMSPropOptimizer(learning_rate,momentum = .8, epsilon = 1e-5)
      train     = optomizer.minimize(loss,var_list = train_vars,global_step=global_step)
  return train

def get_slice(net,num):
  return net[num]

def expert_net(image,label,global_step,training,name = 'ExpertNet',trainable = True):
  with tf.variable_scope(name) as scope:

    image = tf.cast(image,tf.float32)
    image = ops.batch_norm(image,training,trainable = False)

    img_pat  = Image_To_Patch(image)
    lab_pat  = Image_To_Patch(label)

    disc_lab = disc_label_gen(lab_pat)
    disc_log = disc_net(img_pat,training,'disc_net',trainable)

    net1_loss = ops.xentropy_loss(disc_lab,disc_log)
    net1_vars= [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = name + '/disc_net') if var in tf.trainable_variables()]

    patH,patW= ImSizeToPatSize(image)
    log_size = (patH,patW,1)
    neg_log  = lambda: tf.zeros(shape = log_size,dtype = tf.float32)
    rel_log  = seg_net(img_pat,training,'seg_net',trainable)
    seg_log  = []

    # If we should run the second network. Basically, do we 0 out the results
    disc_logit = tf.reshape(tf.equal(tf.argmax(disc_log,-1),1),[disc_log.shape[0].value])
    disc_label = tf.reshape(tf.equal(          disc_lab    ,1),[disc_lab.shape[0].value])

    net2_run = tf.logical_or(tf.logical_and(training,disc_label),tf.logical_and(not training,disc_logit ))

    for x in range(img_pat.shape[0].value):
      pos_log  = lambda: get_slice(rel_log,x)
      seg_log.append(tf.cond(net2_run[x],pos_log,neg_log))
    seg_log = tf.stack(seg_log)

    logit = Patch_To_Image(seg_log)
    net2_loss = ops.log_loss(label,logit)
    net2_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = name + '/seg_net') if var in tf.trainable_variables()]
    
    train = None
    if training:
      train1 = optomize(global_step,net1_loss,net1_vars)
      train2 = optomize(global_step,net2_loss,net2_vars)
      train  = (train1,train2)
    else:
      train = tf.assign_add(global_step,1,name = 'Global_Step')

    # Metrics
    net1_acc = ops.accuracy(disc_label,disc_logit)

    metrics = (net1_acc, net1_loss, net2_loss)

    #Variables
    train_vars = net1_vars + net2_vars
    write_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'batch_norm' in var.name]

    saver_vars = train_vars + write_vars

    return logit,train,metrics,saver_vars


# The discrimatory network.
def disc_net(images,training,name,trainable = True):
  with tf.variable_scope(name) as scope:
    net = images

    strides = factors(100,100)

    for x in range(len(strides)):
      net = ops.dense_block(net,training = training, filters = 4,kernel = 3,  kmap = 3  ,stride = strides[x],trainable = trainable,
                            name = 'DenseBlock%d'%x,prestride_return = False)

    logits = ops.conv2d(net,2,1,stride = 1,name = 'finale',trainable = trainable)
    return logits

# The heuristic network, segemntation.
def seg_net(images,training,name,trainable = True):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    net = images

    net = ops.atrous_block(net,filters = 2, kernel = 4, dilation = 2, kmap = 3, trainable = trainable)

    strides = factors(100,100)
    prestride = []

    for x in range(len(strides)):
      pre, net = ops.dense_block(net,training = training, filters = 4,kernel = 3,
                             kmap = 3, stride = strides[x], trainable = trainable,
                             name = 'DenseBlock%d'%x)
      prestride.append(pre)

    for x in range(len(strides)):
      net = ops.deconvxy(net,training,stride = strides[-(x+1)], trainable = trainable,
                     name = 'DeconvXY%d'%x)
      net = [net,prestride[-(x+1)]]

    net = ops.delist(net)

    logits = ops.conv2d(net,1,1,name = 'finale',trainable = trainable)
    return logits


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
        images, labels, count = inputs(train_run)

      # Our network
      logits,train,metrics,saver_vars = expert_net(images,labels,global_step,train_run)

      # Save operations
      save_imgs  = util.imsave(images,'Image')
      save_logs  = util.imsave(logits,'Logit')
      save_labs  = util.imsave(labels,'Label')
      file_names = ['1img','2log','3labs']
      save_ims   = [save_imgs,save_logs,save_labs]

      # Initialize all variables in network
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())

      # Setting up Tensorboard
      summaries      = tf.summary.merge_all()
      if FLAGS.adv_logging:
        run_options    = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata   = tf.RunMetadata()
      else:
        run_options    = None
        run_metadata   = None

      timestr        = time.strftime("%d_%b_%Y_%H_%M_TRAIN",time.localtime()) if train_run else time.strftime("%d_%b_%Y_%H_%M_TEST",time.localtime())
      # Location to log to.
      filestr        = FLAGS.run_dir + "tensorlogs/" + timestr + "/"
      print("Running from: " + filestr)
      # Location and name to save checkpoint.
      savestr        = FLAGS.run_dir + FLAGS.ckpt_name
      # Location and name to inspect checkpoint for logging.
      logstr         = filestr + 'model.ckpt'
      # Setting up summary writer.
      writer         = tf.summary.FileWriter(filestr,sess.graph)

      # Setting up the checkpoint saver and training coordinator for the network
      saver          = tf.train.Saver(saver_vars)
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
        ops = [train,summaries,metrics] if train_run else [train,summaries,metrics,save_ims]

        step = tf.train.global_step(sess,global_step)
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network and write summaries
          if train_run:
            _,_summ_result,_metrics       = sess.run(ops, options = run_options, run_metadata = run_metadata)
          else:
            _,_summ_result,_metrics,_imgs = sess.run(ops, options = run_options, run_metadata = run_metadata)

          # Write summaries
          if FLAGS.advanced_logging:
            writer.add_run_metadata(run_metadata,'step%d'%step)
          writer.add_summary(_summ_result,step)

          #Write the cmat to a file at each step, write images if testing.
          if not train_run:
            for n in range(len(_imgs)):
              for x in range(FLAGS.batch_size):
                with open(filestr + '%d_%d_' + file_names[n] + '.png'%(step,x),'wb+') as f:
                  f.write(_imgs[n][x])

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
