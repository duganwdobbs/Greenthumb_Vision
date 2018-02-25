
def inputs():
  images = tf.placeholder(tf.float32,(256,256))
  return images

def inference(images,name):
  training = False
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

        # Theoretically, the network will be 8x8x128, for 8192 neurons in the first
        #    fully connected network.
      net = util.squish_to_batch(net)
      _b,neurons = net.get_shape().as_list()

    with tf.variable_scope('Plant_Neurons') as scope:
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

  return p_log,d_log

# Receives an image
def deploy()
