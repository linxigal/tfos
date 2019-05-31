#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/28 8:39
:File       : cnn.py
"""

import tensorflow as tf


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, weights, bias):
    conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + bias)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc(x, weights, bias, keep_prob=None):
    rs = tf.nn.relu(tf.matmul(x, weights) + bias)
    if keep_prob:
        rs = tf.nn.dropout(rs, keep_prob)
    return rs


def softmax(x, weights, bias):
    return tf.nn.softmax(tf.matmul(x, weights) + bias)

