
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

class NetWork(object):
    def __init__(self,gender_dim,age_dim):
        self.gender_dim = gender_dim
        self.age_dim = age_dim
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

    def buildNetWork(self,images,keep_probability,is_train = False,weight_decay =0.0,reuse = None):

        with slim.arg_scope([slim.conv2d,slim.fully_connected],
                            weights_initializer = slim.xavier_initializer(),
                            weights_regularizer = slim.l2_regularizer(weight_decay),
                            normalizer_fn = slim.batch_norm,
                            normalizer_params = self.batch_norm_params):
            with tf.variable_scope('attri',[images],reuse = reuse):

                with slim.arg_scope([slim.batch_norm,slim.dropout],is_training = is_train):
                    with slim.arg_scope([slim.conv2d],stride = 1, padding = 'SAME'):

                        #112*112 * 3
                        net = slim.conv2d(images,32,3,scope='Conv2d_1')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_1')
                        # 56 * 56 * 32
                        net = slim.conv2d(net,64,3,scope='Conv2d_2')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_2')

                        # 28 * 28 * 64
                        net = slim.conv2d(net,64,3,scope='conv2d_3')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_3')

                        #14 * 14 * 128
                        net = slim.conv2d(net,128,3,scope='conv2d_4')
                        net = slim.max_pool2d(net,kernel_size=2,stride=2,padding='VALID',scope='MaxPool_4')
                        # 7 * 7 * 128



                        # net1 = slim.conv2d(net, 128, 3, stride=1, padding='VALID', scope='conv2d_net1_5')
                        net1 = slim.flatten(net)
                        net1 = slim.fully_connected(net1,1024,scope="full_net1_1")
                        net1 = slim.dropout(net1, keep_prob=keep_probability,is_training=is_train)
                        net1 = slim.fully_connected(net1, 512, scope="full_net1_2")
                        net1 = slim.fully_connected(net1,self.gender_dim,activation_fn=None,normalizer_fn = None,
                                                    normalizer_params = None, scope='full_net1_3')


                        # # net2 = slim.conv2d(net,128,3,stride=1,padding='VALID',scope='conv2d_net2_5')
                        # net2 = slim.flatten(net)
                        # net2 = slim.fully_connected(net2,1024,scope='full_net2_1')
                        # net2 = slim.fully_connected(net2, 512, scope='full_net2_2')
                        # # net2 = slim.dropout(net2,keep_prob=keep_probability)
                        # net2 = slim.fully_connected(net2,self.age_dim,activation_fn=None, scope='full_net2_3')

        return  net1







