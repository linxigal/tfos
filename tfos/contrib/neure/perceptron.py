#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/28 8:41
:File       : perceptron.py
"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope


def perceptron(x, y, n_hidden):
    n_input = x.shape(1)
    weights = tf.Variable(tf.random_normal([n_input, n_hidden]))
    biases = tf.Variable(tf.random_normal([n_hidden])),
    out_layer = tf.add(tf.matmul(x, weights), biases)
    return out_layer, y


def multilayer_perceptron(x, y, hidden_units):
    x_ = x
    for layer_id, n_hidden in enumerate(hidden_units):
        with variable_scope.variable_scope(f'hidden_layer_{layer_id}'):
            x_, _ = perceptron(x_, y, n_hidden)
    return x_, y
