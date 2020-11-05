# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v1 network architecture
# --------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from config import model_params

class Network(object):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.classes = model_params['class']
        self.class_num = len(self.classes)
        self.image_size = model_params['image_size']
        self.cell_size = model_params['cell_size']
        self.boxes_per_cell = model_params['boxes_per_cell']
        self.leaky_alpha = model_params['alpha']
        self.keep_prob = model_params['keep_prob']
        self.output_size = self.cell_size * self.cell_size * (5 * self.boxes_per_cell + self.class_num)
        self.class_scale = model_params['class_scale']
        self.object_scale = model_params['object_scale']
        self.noobject_scale = model_params['noobject_scale']
        self.coord_scale = model_params['coord_scale']

    def _leaky_relu(self, alpha):
        def op(inputs):
            return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

    def _build_network(self, inputs, scope='yolo_v1'):
        with tf.name_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=self._leaky_relu(self.leaky_alpha),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
                net = tf.pad(inputs, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = slim.flatten(net, scope='flat_31')
                net = slim.fully_connected(net, 512, scope='fc_32')
                net = slim.fully_connected(net, 4096, scope='fc_33')
                net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_train, scope='dropout_34')
                logits = slim.fully_connected(net, self.output_size, activation_fn=None, scope='fc_35')

        return logits




