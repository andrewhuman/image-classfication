
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

class NetWork(object):
    def __init__(self,gender_dim):
        self.gender_dim = gender_dim
        self.batch_norm_params ={
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        }

    def buildNetWork(self,images,keep_probability=1.0,is_train = False,weight_decay =5e-4,reuse = None):

        with slim.arg_scope([slim.conv2d,slim.fully_connected],
                            weights_initializer = slim.xavier_initializer(),
                            weights_regularizer = slim.l2_regularizer(weight_decay),
                            normalizer_fn = slim.batch_norm,
                            normalizer_params = self.batch_norm_params):
            with tf.variable_scope('attri',[images],reuse = reuse):

                with slim.arg_scope([slim.batch_norm,slim.dropout],is_training = is_train):
                    with slim.arg_scope([slim.conv2d],stride = 1, padding = 'SAME'):

                        #32*32 * 3
                        net = slim.conv2d(images,32,3,scope='Conv2d_1')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_1')

                        # 16 * 16 * 32
                        print('MaxPool_1 = ',net.get_shape().as_list())
                        net = slim.conv2d(net,64,3,scope='Conv2d_2')
                        print('MaxPool_2 = ', net.get_shape().as_list())
                        # 16 * 16 * 64
                        net = slim.conv2d(net,64,3,scope='conv2d_3')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_3')
                        print('MaxPool_3 = ', net.get_shape().as_list())

                        #8 * 8 * 64
                        net = slim.conv2d(net,128,3,scope='conv2d_4')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_4')
                        # 4 * 4 * 128
                        net = slim.conv2d(net, 256, 3, scope='conv2d_5')
                        print('MaxPool_4 = ', net.get_shape().as_list())
                        #4 * 4 * 256

                        net1 = tf.reduce_mean(net,[1,2])

                        print('net1 = ', net1.get_shape().as_list())

                        out = slim.fully_connected(net1,self.gender_dim,activation_fn=None,normalizer_fn = None,
                                                    normalizer_params = None, scope='full_net1_3')
                        print('out = ', out.get_shape().as_list())

                        netup = slim.conv2d(net,128,3,scope='up_conv_5')
                        print('up_conv_5 = ', netup.get_shape().as_list())
                        #4 * 4 * 128

                        netup = tf.image.resize_nearest_neighbor(netup,(8,8))
                        netup = slim.conv2d(netup,64,3,scope='up_conv_4')
                        print('up_conv_4 = ', netup.get_shape().as_list())
                        # 8 * 8 * 64

                        netup = tf.image.resize_nearest_neighbor(netup, (16, 16))
                        netup = slim.conv2d(netup,64,3,scope='up_conv_3')
                        print('up_conv_3 = ', netup.get_shape().as_list())
                        # 16 * 16 * 64

                        netup = slim.conv2d(netup,32,3,scope='up_conv_2')
                        print('up_conv_2 = ', netup.get_shape().as_list())

                        # 16 * 16 * 32

                        netup = tf.image.resize_nearest_neighbor(netup,(32,32))
                        netup = slim.conv2d(netup,3,3,scope='up_conv_1',activation_fn=None,normalizer_params=None,normalizer_fn=None)
                        # netup = tf.nn.sigmoid(netup,name='decoded')
                        # 32 * 32 * 3

                        print('up_conv_1 = ', netup.get_shape().as_list())


        return  out,netup







