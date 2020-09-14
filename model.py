import tensorflow as tf
import tensorflow.contrib.slim as slim

class BIGAN(object):
    def __init__(self, config):
        self.img_width = config.IMG_SIZE
        self.img_height = config.IMG_SIZE
        self.channel = 1
        self.latent_dim  = config.LATENT_DIM
        self.global_step = tf.Variable(initial_value = 0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.image = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.channel])
        #self.image = tf.placeholder(tf.float32, [None, None, None, self.channel])
        self.random_z = tf.placeholder(tf.float32, [None, self.latent_dim])

        self.generated_data = self.generator(self.random_z)
        self.inference_codes = self.encoder(self.image)

        self.real_logits = self.discriminator(self.image, self.inference_codes)
        self.fake_logits = self.discriminator(self.generated_data, self.random_z, reuse=True)

        self.loss_D_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_logits), logits=self.real_logits, scope='loss_D_real')
        self.loss_D_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_logits), logits=self.fake_logits, scope='loss_D_fake')
        with tf.variable_scope('loss_D'):
            self.loss_D = self.loss_D_real+self.loss_D_fake

        self.loss_G_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.real_logits), logits=self.real_logits, scope='loss_G_fake')
        self.loss_G_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_logits), logits=self.fake_logits, scope='loss_G_real')
        with tf.variable_scope('loss_G'):
            self.loss_G = self.loss_G_real+self.loss_G_fake

        self.real_sample_data = self.image
        self.sample_inference_codes = self.encoder(self.real_sample_data, is_training=False, reuse=True)
        self.reconstruction_data = self.generator(self.sample_inference_codes, is_training=False, reuse=True)
        self.generated = self.generator(self.random_z, is_training=False, reuse=True)

        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

        tf.summary.scalar('losses/loss_Discriminator', self.loss_D)
        tf.summary.scalar('losses/loss_Generator', self.loss_G)

        for var in self.D_vars:
            tf.summary.histogram(var.op.name, var)
        for var in self.G_vars:
            tf.summary.histogram(var.op.name, var)

        tf.summary.image('random_images', self.generated_data, max_outputs=4)

        print("Done building")
        
        
    def generator(self, random_z, is_training=True, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}

            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[4,4],
                                stride=[2,2],
                                normalizer_fn=None,
                                normalizer_params=None):
                                #normalizer_fn=slim.batch_norm,
                                #normalizer_params=batch_norm_params):

                net = tf.reshape(random_z, [-1, 1, 1, self.latent_dim])
                net = slim.conv2d_transpose(net, 512, kernel_size=[3,3], padding='VALID')
                net = slim.conv2d_transpose(net, 256, kernel_size=[3,3], padding='VALID')
                #net = slim.conv2d_transpose(net, 256, kernel_size=[3,3], padding='VALID')
                net = slim.conv2d_transpose(net, 128, kernel_size=[3,3], padding='VALID')
                #net = slim.conv2d_transpose(net, 128, kernel_size=[3,3], padding='VALID')
                net = slim.conv2d_transpose(net, 64, kernel_size=[3,3], padding='VALID')
                net = slim.conv2d_transpose(net, 64, padding='VALID')
                net = slim.conv2d_transpose(net, 32)
                net = slim.conv2d_transpose(net, 1, normalizer_fn=None, activation_fn=tf.nn.tanh)
                #print(net)
                return net

        
    def encoder(self, data, is_training=True, reuse=False):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}

            with slim.arg_scope([slim.conv2d],
                                kernel_size=[4,4],
                                stride=[2,2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=None,
                                normalizer_params=None):

                net = slim.conv2d(data, 32, normalizer_fn=None, scope='encoder_layer1')
                net = slim.conv2d(net, 64, scope='encoder_layer2')
                net = slim.conv2d(net, 64, kernel_size=[3,3], padding='VALID', scope='encoder_layer3')
                #net = slim.conv2d(net, 128, kernel_size=[3,3], padding='VALID', scope='encoder_layer4')
                net = slim.conv2d(net, 128, kernel_size=[3,3], padding='VALID', scope='encoder_layer5')
                #net = slim.conv2d(net, 256, kernel_size=[3,3], padding='VALID', scope='encoder_layer6')
                net = slim.conv2d(net, 256, kernel_size=[3,3], padding='VALID', scope='encoder_layer7')
                net = slim.conv2d(net, 512, kernel_size=[3,3], padding='VALID', scope='encoder_layer8')
                #net = slim.conv2d(net, 512, kernel_size=[3,3], padding='VALID', scope='encoder_layer9')
                #print(net)
                net = slim.conv2d(net, self.latent_dim, kernel_size=[3,3], stride=[1,1], padding='VALID', normalizer_fn=None, activation_fn=None, scope='layer9')
                #print(net)
                z_hat = tf.squeeze(net, axis=[1,2])                
                return z_hat

    def discriminator(self, data, latent_code, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                'epsilon': 0.001,
                                'scope': 'batch_norm'}

            with slim.arg_scope([slim.conv2d],
                            kernel_size=[4,4],
                            stride=[2,2],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=None,
                            normalizer_params=None):

                fc = slim.fully_connected(latent_code, self.img_height*self.img_width*2)
                fc_reshape = tf.reshape(fc, [-1, self.img_height, self.img_width, 2])

                #print(data)
                #print(fc_reshape)
                inputs = tf.concat([data, fc_reshape], axis=3)
                layer = slim.conv2d(inputs, 32, normalizer_fn=None, scope='discriminator_layer1')
                layer = slim.conv2d(layer, 64, scope='discriminator_layer2')
                layer = slim.conv2d(layer, 64, scope='discriminator_layer3')
                #layer = slim.conv2d(layer, 128, scope='discriminator_layer4')
                layer = slim.conv2d(layer, 128, scope='discriminator_layer5')
                #layer = slim.conv2d(layer, 256, scope='discriminator_layer6')
                layer = slim.conv2d(layer, 256, scope='discriminator_layer7')
                print(layer)
                layer = slim.conv2d(layer, 512, kernel_size=[3,3], padding='VALID', scope='discriminator_layer8')
                print(layer)
                layer = slim.conv2d(layer, 1, kernel_size=[3,3], stride=[1,1], padding='VALID', normalizer_fn=None, activation_fn=None, scope='discriminator_layer9')
                print(layer)
                logits = tf.squeeze(layer, axis=[2,3])

                return logits