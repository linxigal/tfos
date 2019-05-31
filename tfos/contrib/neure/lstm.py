#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/28 8:39
:File       : lstm.py
"""

import tensorflow as tf


def lstm_cell(i, o, state, shapes):
    """

    :param i:
    :param o:
    :param state:
    :param shapes: must be two dimensions, first is input dimension, second is hidden cell dimension
    :return:
    """
    if len(shapes) != 2:
        raise ValueError("shapes of parameter must be two dimensions!")

    # Input gate: input, previous output, and bias
    ix = tf.get_variable(tf.truncated_normal(shapes), name='ix')
    im = tf.get_variable(tf.truncated_normal(shapes), name='im')
    ib = tf.get_variable(tf.zeros([1, shapes[1]]), name='ib')
    # Forget gate: input, previous output, and bias
    fx = tf.get_variable(tf.truncated_normal(shapes), name='fx')
    fm = tf.get_variable(tf.truncated_normal(shapes), name='fm')
    fb = tf.get_variable(tf.zeros([1, shapes[1]]), name='fb')
    # Memory cell: input, state, and bias
    cx = tf.get_variable(tf.truncated_normal(shapes), name='cx')
    cm = tf.get_variable(tf.truncated_normal(shapes), name='cm')
    cb = tf.get_variable(tf.zeros([1, shapes[1]]), name='cb')
    # Output gate: input, previous output, and bias
    ox = tf.get_variable(tf.truncated_normal(shapes), name='ox')
    om = tf.get_variable(tf.truncated_normal(shapes), name='om')
    ob = tf.get_variable(tf.zeros([1, shapes[1]]), name='ob')
    #  lstm internal cell calculate
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)

    state = forget_gate * state + input_gate * update
    return output_gate * tf.tanh(state), state



