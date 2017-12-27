
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def resnet_conv(net,kernel_num,scale = 1.0,activation_fn = tf.nn.relu,scope = None,reuse=None):
    with tf.variable_scope(scope,'resnet_conv',[net],reuse=reuse):
        up = slim.conv2d(net,kernel_num,3,scope='conv2d_a_1x1')
        print("resnet_conv  conv2d_a_1x1 ",net.get_shape().as_list())

        up = slim.conv2d(up,kernel_num,3,activation_fn=None,scope='conv2d_b_1x1')
        print("resnet_conv  conv2d_b_1x1 ",net.get_shape().as_list())
        net += up * scale
        if activation_fn:
            net = activation_fn(net)
    return net
    
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.5,
                        bottleneck_layer_size=512,
                        reuse=None, 
                        scope='InceptionResnetV1'):
    """Creates model
    Args:
      inputs: a 4-D tensor of size [batch_size, 32, 32, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                print("inputs ",inputs.get_shape().as_list())

                #15, 15, 64
                net = slim.conv2d(inputs,64,3,stride=2,padding='VALID',scope='conv_1_3x3')
                print("after conv_1_3x3 ",net.get_shape().as_list())

                #15, 15, 64
                net = slim.conv2d(net,64,3,scope='conv_2_3x3')
                print("after conv_2_3x3 ",net.get_shape().as_list())

                net = resnet_conv(net,64)
                print("after resnet_conv 1 ",net.get_shape().as_list())
                net = resnet_conv(net,64)
                print("after resnet_conv 2 ",net.get_shape().as_list())

                #15, 15, 128
                net = slim.conv2d(net,128,1,scope='conv_3_1x1')
                print("after conv_3_1x1 ",net.get_shape().as_list())
                net = resnet_conv(net,128)
                print("after resnet_conv 4 ",net.get_shape().as_list())
                net = resnet_conv(net,128)
                print("after resnet_conv 5 ",net.get_shape().as_list())

                #7x7x128
                net = slim.conv2d(net,256,3,stride=2,padding='VALID',scope='conv_4_3x3')
                print("after conv_4_3x3 ",net.get_shape().as_list())
                #7x7x256
                net = resnet_conv(net,256)
                print("after resnet_conv 6 ",net.get_shape().as_list())
                net = resnet_conv(net,256)
                print("after resnet_conv 7 ",net.get_shape().as_list())
                net = resnet_conv(net,256)
                print("after resnet_conv 8 ",net.get_shape().as_list())

                #3x3x256
                net = slim.conv2d(net,512,3,stride=2,padding='VALID',scope='conv_5_3x3')
                print("after conv_5_3x3  ",net.get_shape().as_list())

                net = tf.nn.relu(net)
                #3x3x512
                net = resnet_conv(net,512)
                print("after resnet_conv 9 ",net.get_shape().as_list())
                net = resnet_conv(net,512)
                print("after resnet_conv 10 ",net.get_shape().as_list())

                with tf.variable_scope('Logits'):

                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                #     net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                #                        scope='Dropout')
                #
                #     end_points['PreLogitsFlatten'] = net
                #
                # net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                #         scope='Bottleneck', reuse=False)
                print("after fully_connected ",net.get_shape().as_list())
    return net, end_points

