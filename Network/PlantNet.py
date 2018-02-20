import os
import time
import platform

import numpy         as     np
import tensorflow    as     tf
import spec_ops      as     stf

from multiprocessing import Process
from inspector       import inspect
from util            import factors
from util            import ImageToPatch
from util            import PatchToImage
from tfrecord        import inputs
from tfrecord        import sizes

imgW,imgH,save_stride = sizes()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS

if   platform.system() == 'Windows':
  flags.DEFINE_string ('base_dir'  ,'D:/Greenthumb_Vision/','Base os specific DIR')
elif platform.system() == 'Linux':
  flags.DEFINE_string ('base_dir'  ,'/home/ddobbs/Greenthumb_Vision/','Base os specific DIR')

flags.DEFINE_boolean('l2_loss'            ,False ,'If we use l2 regularization')
flags.DEFINE_boolean('batch_norm'         ,True  ,'If we use batch normalization')
flags.DEFINE_boolean('lr_decay'           ,True  ,'If we use Learning Rate Decay')
flags.DEFINE_boolean('advanced_logging'   ,False ,'If we log metadata and histograms')
flags.DEFINE_integer('num_epochs'         ,None  ,'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size'         ,1     ,'Batch size for training.')
flags.DEFINE_integer('train_steps'        ,10000 ,'Number of steps for training on counting')
flags.DEFINE_integer('num_plant_classes'  ,10    ,'# Classes')
flags.DEFINE_integer('num_disease_classes',10    ,'# Classes')
flags.DEFINE_string ('run_dir'    , FLAGS.base_dir + '/network_log/','Location to store the Tensorboard Output')
flags.DEFINE_string ('train_dir'  ,FLAGS.base_dir  + '/'            ,'Location of the tfrecord files.')
flags.DEFINE_string ('ckpt_name'  ,'cows.ckt'                       ,'Checkpoint name')
flags.DEFINE_string ('ckpt_i_name','cows-interrupt.ckpt'            ,'Checkpoint Interrupt name')
flags.DEFINE_string ('net_name','PlantVision'                       ,'Network name')

def launchTensorBoard(directory):
  sleep(30)
  os.system(directory)

def training(global_step,loss,train_vars,learning_rate = .001):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.variable_scope("Optimizer") as scope:
      if FLAGS.lr_decay:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   250, 0.85, staircase=True)
        tf.summary.scalar("Learning_Rate",learning_rate)

      optomizer = tf.train.RMSPropOptimizer(learning_rate,momentum = .8, epsilon = 1e-5)
      train     = optomizer.minimize(loss,var_list = train_vars,global_step=global_step)

      return train

# The segmentation network.
def inference(images,p_lab,d_lab,training,name,trainable = True):
  with tf.variable_scope(name) as scope:
    net = images
    net = delist(net)
    net = tf.cast(net,tf.float32)
    net = batch_norm(net,training,trainable)

    net = stf.dense_reduction(net,training,trainable,...,...)
    net = stf.dense_reduction(net,training,trainable,...,...)
    net = stf.dense_reduction(net,training,trainable,...,...)
    net = stf.resnet_a(net)
    net = stf.resnet_b(net)
    net = stf.resnet_reduction(net)
    net = stf.resnet_b(net)
    net = stf.resnet_reduction(net)
    net = stf.resnet_a(net)
    net = stf.conv2d(net,128,1)

    net = stf.ifft2d(net)

    net = util.squish_to_batch(net)
    _b,neurons = net.get_shape().as_list()

    net = tf.layers.dense(net,neurons)

    p_log      = tf.layers.dense(net,FLAGS.num_plant_classes)

    d_net = []
    for x in range(FLAGS.num_plant_classes):
      chan = tf.layers.dense(net,FLAGS.num_disease_classes)
      d_net.append(chan)
    d_net = delist(d_net)

    if training:
      index = d_label
    else:
      index = tf.argmax(p_log)

    d_log      = d_net[index]

    p_loss,p_metrics = metrics(labels = p_lab,logits = p_log,name = "Plant_Metrics")
    d_loss,d_metrics = metrics(labels = d_lab,logits = d_log,name = "Disease_Metrics")
    metrics = (p_metrics,d_metrics)

    loss = p_loss + d_loss

    return p_logit,d_logit,loss,metrics

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
        images, p_lab, d_lab = inputs(train=train_run,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs,num_classes = FLAGS.num_classes)

      p_log,d_log,loss,metrics = p_inference(images,p_lab,d_lab,training = train_run,name = FLAGS.net_name,trainable = True)

      train_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = FLAGS.net_name) if var in tf.trainable_variables()]
      write_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'batch_norm' in var.name]
      if train_run:
        train = training(global_step,loss,train_vars)
      else:
        train = tf.assign_add(global_step,1,name = 'Global_Step')

      # Save operations
      save_images = imsave(images,names = 'Images')

      save_imgs = [save_images]

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
        ops = [train,summaries,metrics] if train_run else [train,summaries,metrics,save_imgs,p_lab,p_log,d_lab,d_log]

        step = tf.train.global_step(sess,global_step)
        while not coord.should_stop() and step <= FLAGS.train_steps:
          step = tf.train.global_step(sess,global_step)

          # Run the network and write summaries
          if train_run:
            _,_summ_result,_metrics       = sess.run(ops, options = run_options, run_metadata = run_metadata)
          else:
            _,_summ_result,_metrics,_imgs,_p_lab,_p_log,_d_lab,_d_log = sess.run(ops, options = run_options, run_metadata = run_metadata)

          # Write summaries
          if FLAGS.advanced_logging:
            writer.add_run_metadata(run_metadata,'step%d'%step)
          writer.add_summary(_summ_result,step)

          #Write the cmat to a file at each step, write images if testing.
          if not train_run:
            for x in range(len(_imgs)):
              for d in range(FLAGS.batch_size):
                with open(filestr + '%d_plant_%d_%d_disease_%d_%d'%(step,x,_p_lab[d],_p_log[d],_d_lab[d],_d_log[d]) + im_t[x] + '.png','wb+') as f:
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
