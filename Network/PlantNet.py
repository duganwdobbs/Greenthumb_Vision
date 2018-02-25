import os
import time
import platform
import ops
import util

import spec_ops      as stf

import numpy         as     np
import tensorflow    as     tf
import spec_ops      as     stf

from multiprocessing import Process
from inspector       import inspect
from tfrecord        import inputs
from tfrecord        import sizes

imgW,imgH = sizes()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'E:/Greenthumb_Vision'          ,'Base os specific DIR')
  flags.DEFINE_string ('log_dir'   ,'F:/Greenthumb_Vision'          ,'Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss'            ,True  ,'If we use l2 regularization')
flags.DEFINE_boolean('batch_norm'         ,True  ,'If we use batch normalization')
flags.DEFINE_boolean('lr_decay'           ,True  ,'If we use Learning Rate Decay')
flags.DEFINE_boolean('advanced_logging'   ,False ,'If we log metadata and histograms')
flags.DEFINE_boolean('log_imgs'           ,False ,'If we log images to tfrecord')


flags.DEFINE_integer('num_epochs'         ,1     ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size'         ,4     ,'Batch size for training.')
flags.DEFINE_integer('train_steps'        ,100000,'Number of steps for training on counting')
flags.DEFINE_integer('num_plant_classes'  ,10    ,'# Classes')
flags.DEFINE_integer('num_disease_classes',10    ,'# Classes')


flags.DEFINE_string ('run_dir'    , FLAGS.log_dir + '/network_log/','Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  , FLAGS.base_dir                 ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  ,'greenthumb.ckpt'               ,'Checkpoint name')
flags.DEFINE_string ('net_name'   ,'PlantVision'                   ,'Network name')

def trainer(global_step,loss,train_vars,learning_rate = .001):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.variable_scope("Optimizer") as scope:
      if FLAGS.lr_decay:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   2500, 0.85, staircase=True)
        tf.summary.scalar("Learning_Rate",learning_rate)

      optomizer = tf.train.AdamOptimizer(learning_rate,epsilon = 1e-5)
      train     = optomizer.minimize(loss,var_list = train_vars)

      return train

# The segmentation network.
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

    with tf.variable_scope('Plant_Neurons') as scope:
      # Theoretically, the network will be 8x8x128, for 8192 neurons in the first
      #    fully connected network.
      net = util.squish_to_batch(net)
      _b,neurons = net.get_shape().as_list()

      # Fully connected network with number of neurons equal to the number of
      #    parameters in the network at this point
      net = tf.layers.dense(net,neurons)

      # Fully connected layer to extract the plant classification
      #    NOTE: This plant classification will be used to extract the proper
      #          disease classification matrix
      p_log      = tf.layers.dense(net,FLAGS.num_plant_classes)

    with tf.variable_scope('Disease_Neurons') as scope:
      # Construct a number of final layers for diseases equal to the number of
      # plants.
      d_net = []
      for x in range(FLAGS.num_plant_classes):
        chan = tf.layers.dense(net,FLAGS.num_disease_classes,name = 'Disease_%d'%x)
        d_net.append(chan)
      d_net = tf.stack(d_net)
      d_log = d_net

    return p_log,d_log

def metrics(global_step,p_lab,d_lab,training):
  with tf.variable_scope('Formatting') as scope:
    # If we're training, we want to not use the plant network output, rather the
    #    plant label. This ensures that the disease layers train properly.
    #    NOTE: The disease loss function only trains based on this final layer.
    #          IE: The disease gradient does not flow through the whole network,
    #              using the plant network as its preprocessing.
    index = d_lab #if training else tf.argmax(p_log)
    index = tf.cast(index,tf.int32)

    size = [1,1,FLAGS.num_disease_classes]
    # Extract the disease logit per example in batch
    d_log = []
    for x in range(FLAGS.batch_size):
      start = [index[x],x,0]
      val = tf.slice(d_net,start,size)
      d_log.append(val)
    d_log = tf.stack(d_log)
    d_log = tf.reshape(d_log,[FLAGS.batch_size,FLAGS.num_disease_classes])

  # Get the losses and metrics

  with tf.variable_scope('Metrics') as scope:
    p_vars = [var for var in tf.trainable_variables() if 'Process_Network' in var.name or 'Plant_Neurons' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'Disease_Neurons' in var.name]

    p_loss = ops.xentropy_loss(p_lab,p_log,p_vars,name = "Plant_Loss")
    d_loss = ops.xentropy_loss(d_lab,d_log,d_vars,name = "Disease_Loss")
    p_log = tf.argmax(p_log,-1)
    d_log = tf.argmax(d_log,-1)
    p_acc  = ops.accuracy(p_lab,p_log,name = 'Plant_Accuracy')
    d_acc  = ops.accuracy(d_lab,d_log,name = 'Disease_Accuracy')

    metrics = (p_loss,d_loss,p_acc,d_acc)

  with tf.variable_scope('Trainer') as scope:
    train = tf.assign_add(global_step,1,name = 'Global_Step')
    if training:
      p_train = trainer(global_step,p_loss,p_vars)
      d_train = trainer(global_step,d_loss,d_vars)
      train   = (train,p_train,d_train)

  return d_log,train,metrics

# Runs the tape training.
def train(train_run = True, restore = False):
  with tf.Graph().as_default():
      config         = tf.ConfigProto(allow_soft_placement = True)
      config.gpu_options.allow_growth = True
      sess           = tf.Session(config = config)
      # Setting up a new session
      global_step    = tf.Variable(1,name='global_step',trainable=False)

      if not train_run:
        FLAGS.batch_size = 1
        FLAGS.num_epochs = 1

      # Build the network from images, inference, loss, and backpropogation.
      with tf.variable_scope("Net_Inputs") as scope:
        images, p_lab, d_lab = inputs(global_step,train=train_run,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
        p_lab = tf.reshape(p_lab,[FLAGS.batch_size])
        d_lab = tf.reshape(d_lab,[FLAGS.batch_size])

      p_log,d_log         = inference(images,training = train_run,name = FLAGS.net_name,trainable = True)
      # d_log at this point is full [10][batch][10], metrics formats it correctly.
      d_log,train,metrics = metrics(global_step,p_lab,d_log,d_lab,training = train_run)

      b_norm_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'batch_norm' in var.name]

      # Save operations
      save_images = util.imsave(images,name = 'Images')
      save_imgs = [save_images]
      im_t      = ['Images']

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
      if not train_run:
        savestr = logstr
      # Setting up summary writer.
      writer         = tf.summary.FileWriter(filestr,sess.graph)

      # Setting up the checkpoint saver and training coordinator for the network
      saver_vars = [var for var in tf.trainable_variables() if var not in b_norm_vars] + b_norm_vars
      if(restore or not train_run):
        # inspect(tf.train.latest_checkpoint(FLAGS.run_dir))
        if restore:
          restore_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Optimizer' not in var.name]
          saver = tf.train.Saver(restore_vars)
        else:
          saver = tf.train.Saver(saver_vars)

        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.run_dir))
        if not train_run:
          # Save the loaded testing checkpoint in the log folder.
          saver.save(sess,logstr)
      saver     = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

      # Starts the input generator
      coord          = tf.train.Coordinator()
      threads        = tf.train.start_queue_runners(sess = sess, coord = coord)

      try:
        ops = [train,summaries,metrics,p_lab,p_log,d_lab,d_log] if train_run else [train,summaries,metrics,save_imgs,p_lab,p_log,d_lab,d_log]

        step = tf.train.global_step(sess,global_step)
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network and write summaries
          if train_run:
            _,_summ_result,_metrics,_p_lab,_p_log,_d_lab,_d_log       = sess.run(ops, options = run_options, run_metadata = run_metadata)
          else:
            _,_summ_result,_metrics,_imgs,_p_lab,_p_log,_d_lab,_d_log = sess.run(ops, options = run_options, run_metadata = run_metadata)

          # Some basic label / logit output
          # for d in range(FLAGS.batch_size):
            # print("Label / Prediciton Plant: %d / %d Disease: %d / %d"%(_p_lab[d],_p_log[d],_d_lab[d],_d_log[d]))

          # Write summaries
          if FLAGS.advanced_logging:
            writer.add_run_metadata(run_metadata,'step%d'%step)
          writer.add_summary(_summ_result,step)

          #Write the cmat to a file at each step, write images if testing.
          if not train_run and step % 1000 == 0:
            for x in range(len(_imgs)):
              for d in range(FLAGS.batch_size):
                with open(filestr + '%d_%d_plant_%d_%d_disease_%d_%d'%(step,d,_p_lab[d],_p_log[d],_d_lab[d],_d_log[d]) + '_img.png','wb+') as f:
                  f.write(_imgs[x][d])

          if step%100 == 0 and train_run:
            # Save some checkpoints
            saver.save(sess,savestr,global_step = step)
            saver.save(sess,logstr,global_step = step)
      except KeyboardInterrupt:
        if train_run:
          saver.save(sess,savestr,global_step = step)
          saver.save(sess,logstr,global_step = step)
      except tf.errors.OutOfRangeError:
        saver.save(sess,savestr,global_step = step)
        saver.save(sess,logstr,global_step = step)
        print("Sumthin messed up man.")
      finally:
        if train_run:
          saver.save(sess,savestr)
          saver.save(sess,logstr)
        coord.request_stop()
      coord.join(threads)
      sess.close()

def main(_):
  for epoch in range(2,10):
    train(train_run = True,  restore = (epoch != 0) )
    train(train_run = False, restore = False)
    print("Epoch %d training+validation complete"%epoch+1)

if __name__ == '__main__':
  tf.app.run()
