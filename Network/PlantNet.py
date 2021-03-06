import os
import time
import platform
import ops
import util

import tensorflow    as     tf
import spec_ops      as     stf

import resnet_modules
resnetA = resnet_modules.block35
resnetB = resnet_modules.block17
resnetC = resnet_modules.block8

from multiprocessing import Process
from tfrecord        import inputs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

plants   = [ 'Apple','Cherry','Corn','Grape','Peach','Strawberry','Bell_Pepper','Potato','Tomato']

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'E:/Greenthumb_Vision'          ,'Base os specific DIR')
  flags.DEFINE_string ('log_dir'   ,'F:/Greenthumb_Vision'          ,'Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss'            ,False ,'If we use l2 regularization')
flags.DEFINE_boolean('batch_norm'         ,True  ,'If we use batch normalization')
flags.DEFINE_boolean('advanced_logging'   ,False ,'If we log metadata and histograms')
flags.DEFINE_boolean('log_imgs'           ,False ,'If we log images to tfrecord')


flags.DEFINE_integer('num_epochs'         ,1     ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size'         ,6     ,'Batch size for training.')
flags.DEFINE_integer('train_steps'        ,100000,'Number of steps for training on counting')
flags.DEFINE_integer('num_plant_classes'  ,9     ,'# Classes')
flags.DEFINE_integer('num_disease_classes',10    ,'# Classes')


flags.DEFINE_string ('run_dir'    , FLAGS.log_dir + '/network_log/','Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  , FLAGS.base_dir                 ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  ,'greenthumb.ckpt'               ,'Checkpoint name')
flags.DEFINE_string ('net_name'   ,'PlantVision'                   ,'Network name')

def trainer(global_step,loss,train_vars,fancy = True,learning_rate = .0001):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.variable_scope("Optimizer") as scope:
      if fancy:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   4000, 0.90, staircase=True)
        tf.summary.scalar("Learning_Rate",learning_rate)

      optomizer = tf.train.AdamOptimizer(learning_rate,epsilon = 1e-5)
      train     = optomizer.minimize(loss,var_list = train_vars)

      return train

# The Classification network.
def inference(images,training,name,trainable = True):
  ops.init_scope_vars()
  with tf.variable_scope(name) as scope:
    # Preform some image preprocessing
    net = images
    net = ops.delist(net)
    net = tf.cast(net,tf.float32)
    net = ops.im_norm(net)

    # If we receive image channels in a format that shouldn't be normalized,
    #   that goes here.

    with tf.variable_scope('Process_Network'):
      # Run two dense reduction modules to reduce the number of parameters in
      #   the network and for low parameter feature extraction.
      net = ops.dense_reduction(net,training,filters = 4 , kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_1')
      net = ops.dense_reduction(net,training,filters = 8, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_2')
      net = ops.dense_reduction(net,training,filters = 12, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Block_3')

      # Run the network over some resnet modules, including reduction
      #   modules in order to further reduce the parameters and have a powerful,
      #   proven network architecture.
      net = resnetA(net)
      net = resnetA(net)
      net = ops.dense_reduction(net,training,filters = 12, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Redux_4')

      net = resnetA(net)
      net = resnetA(net)
      net = ops.dense_reduction(net,training,filters = 12, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Redux_5')


      net = resnetB(net)
      net = resnetB(net)
      net = ops.dense_reduction(net,training,filters = 12, kernel = 3, stride = 2,
                                activation=tf.nn.leaky_relu,trainable=trainable,
                                name = 'Dense_Redux_6')


      net = resnetC(net)
      net = resnetC(net)
      # net = ops.dense_reduction(net,training,filters = 56, kernel = 3, stride = 2,
      #                           activation=tf.nn.leaky_relu,trainable=trainable,
      #                           name = 'Dense_Block_7')


    with tf.variable_scope('Plant_Neurons') as scope:
      p_net = util.squish_to_batch(net)
      _b,neurons = net.get_shape().as_list()
      p_log = tf.layers.dense(p_net,neurons,name = 'Plant_Neurons')
      p_log = tf.layers.dense(p_log,FLAGS.num_plant_classes,name = 'Plant_Decider')
      p_log = tf.squeeze(p_log)
    with tf.variable_scope('Disease_Neurons') as scope:
      # A specific disease ResnetC module
      net = resnetC(net)
      net = util.squish_to_batch(net)
      _b,neurons = net.get_shape().as_list()
      # Construct a number of final layers for diseases equal to the number of plants.
      d_log = []
      chan = tf.layers.dense(net,neurons, name = 'Disease_Neurons')
      for x in range(FLAGS.num_plant_classes):
        d_n = tf.layers.dense(chan,FLAGS.num_disease_classes,name = '%s_Decider'%plants[x])
        d_n = tf.squeeze(d_n)
        d_log.append(d_n)
      with tf.variable_scope('DieaseFormatting') as scope:
        d_log = tf.stack(d_log)

    return p_log,d_log

def per_cat_acc(p_lab,d_lab,p_log,d_log):
  with tf.variable_scope("Per_Class_Metrics") as scope:
    oh_p_lab = tf.one_hot(p_lab)
    oh_d_lab = tf.one_hot(d_lab)
    oh_p_log = tf.one_hot(p_log)
    oh_d_log = tf.one_hot(d_log)

    with tf.variable_scope("Sum_Ops") as scope:
      sum_p_lab = tf.zeros(tf.int64,(FLAGS.num_plant_classes))
      sum_d_lab = tf.zeros(tf.int64,(FLAGS.num_plant_classes))
      sum_p_log = tf.zeros(tf.int64,(FLAGS.num_plant_classes))
      sum_d_log = tf.zeros(tf.int64,(FLAGS.num_plant_classes))

      op1       = tf.assign_add(sum_p_lab,oh_p_lab)
      op2       = tf.assign_add(sum_d_lab,oh_d_lab)
      op3       = tf.assign_add(sum_p_log,oh_p_log)
      op4       = tf.assign_add(sum_d_log,oh_d_log)
      ops       = (op1,op2,op3,op4)

    for plant_it in range(FLAGS.num_plant_classes):
      p_acc = sum_p_log[x]
      tf.summary.scalar


def build_metrics(global_step,p_lab,d_lab,p_log,d_logs,training):
  with tf.variable_scope('Formatting') as scope:
    # If we're training, we want to not use the plant network output, rather the
    #    plant label. This ensures that the disease layers train properly.
    #    NOTE: The disease loss function only trains based on this final layer.
    #          IE: The disease gradient does not flow through the whole network,
    #              using the plant network as its preprocessing.
    index = p_lab #if training else tf.argmax(p_log)
    index = tf.cast(index,tf.int32)

    size = [1,1,FLAGS.num_disease_classes]
    # Extract the disease logit per example in batch
    d_log = []
    d_logs = tf.reshape(d_logs,[FLAGS.num_plant_classes,FLAGS.batch_size,FLAGS.num_disease_classes])
    for x in range(FLAGS.batch_size):
      start = [index[x],x,0]
      val = tf.slice(d_logs,start,size)
      d_log.append(val)
    d_log = tf.stack(d_log)
    d_log = tf.reshape(d_log,[FLAGS.batch_size,FLAGS.num_disease_classes])

  # Get the losses and metrics

  with tf.variable_scope('Metrics') as scope:
    # Seperate the variables out of each network, so that we do not train the
    # process network on disease loss.
    p_vars = [var for var in tf.trainable_variables() if 'Process_Network' in var.name or 'Plant_Neurons' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'Disease_Neurons' in var.name]

    # Calculate the losses per network
    p_log = tf.reshape(p_log,[FLAGS.batch_size,FLAGS.num_plant_classes])
    p_loss = ops.xentropy_loss(p_lab,p_log,p_vars,l2 = True ,name = "Plant_Loss")
    d_loss = ops.xentropy_loss(d_lab,d_log,d_vars,l2 = False,name = "Disease_Loss")

    # Flatten the logits so that we only have one output instead of 10
    p_log = tf.argmax(p_log,-1)
    d_log = tf.argmax(d_log,-1)

    # Log the accuracy
    p_acc  = ops.accuracy(p_lab,p_log,name = 'Plant_Accuracy')
    d_acc  = ops.accuracy(d_lab,d_log,name = 'Disease_Accuracy')

    # Create a variable in order to get these operations out of the network
    metrics = (p_loss,d_loss,p_acc,d_acc)

  # Setup the trainer
  with tf.variable_scope('Trainer') as scope:
    train = tf.assign_add(global_step,1,name = 'Global_Step')
    if training:
      p_train = trainer(global_step,p_loss,p_vars,fancy = True ,learning_rate = 1e-5)
      d_train = trainer(global_step,d_loss,d_vars,fancy = False,learning_rate = 1e-4)
      train   = (train,p_train,d_train)

  return p_log,d_log,train,metrics

# Runs the training.
def train(train_run = True, restore = False):
  with tf.Graph().as_default():
      config         = tf.ConfigProto(allow_soft_placement = True)
      config.gpu_options.allow_growth = True
      sess           = tf.Session(config = config)
      # Setting up a new session
      global_step    = tf.Variable(1,name='global_step',trainable=False)

      if not train_run:
        FLAGS.batch_size = 1
      if train_run:
        FLAGS.batch_size = 4

      # Build the network from images, inference, loss, and backpropogation.
      with tf.variable_scope("Net_Inputs") as scope:
        images, p_lab, d_lab = inputs(global_step,train=train_run,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
        p_lab = tf.reshape(p_lab,[FLAGS.batch_size])
        d_lab = tf.reshape(d_lab,[FLAGS.batch_size])

      p_logs,d_logs             = inference(images,training = train_run,name = FLAGS.net_name,trainable = True)
      # d_log at this point is full [10][batch][10], metrics formats it correctly.
      p_log,d_log,train,metrics = build_metrics(global_step,p_lab,d_lab,p_logs,d_logs,training = train_run)

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
        ops = [train,summaries,metrics,p_lab,p_log,d_lab,d_log] if train_run else [train,summaries,metrics,save_imgs,p_lab,p_log,d_lab,d_log,p_logs,d_logs]

        step = tf.train.global_step(sess,global_step)
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network and write summaries
          if train_run:
            _,_summ_result,_metrics,_p_lab,_p_log,_d_lab,_d_log       = sess.run(ops, options = run_options, run_metadata = run_metadata)
          else:
            _,_summ_result,_metrics,_imgs,_p_lab,_p_log,_d_lab,_d_log,_p_logs,_d_logs = sess.run(ops, options = run_options, run_metadata = run_metadata)
            # print("Plant")
            # print(_p_logs)
            # print("Disease")
            # print(_d_logs)

          # Some basic label / logit output
          # for d in range(FLAGS.batch_size):
          #   print("Label / Prediciton Plant: %d / %d Disease: %d / %d"%(_p_lab[d],_p_log[d],_d_lab[d],_d_log[d]))

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
        if train_run:
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
  for epoch in range(0,3):
    # Train a network for 1 epoch
    train(train_run = True,  restore = epoch != 0 )
    # Run validation
    train(train_run = False, restore = False)
    print("Epoch %d training+validation complete"%(epoch+1))

if __name__ == '__main__':
  tf.app.run()
