#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/23 17:11
:File       : regression.py
"""

import tensorflow as tf


def logistic_regression(graph, x, y):
    with graph.as_default():
        features = x.shape[1]
        labels = y.shape[1]
        W = tf.Variable(tf.truncated_normal([features, labels], stddev=1), name="weight")
        b = tf.Variable(tf.zeros([labels]), name="bias")
        # Construct model
        y_ = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax
        pred = tf.argmax(y, 1, name="prediction")
        return y_, pred


def linear_regression(graph, x, y):
    with graph.as_default():
        features = x.shape[1]
        labels = y.shape[1]
        W = tf.Variable(tf.truncated_normal([features, labels], stddev=1), name="weight")
        b = tf.Variable(tf.zeros([labels]), name="bias")
        # Construct a linear model
        pred = tf.add(tf.multiply(x, W), b)
        return pred


def logistic_regression_layer(x, y):
    # Set model weights
    W = tf.Variable(tf.truncated_normal([4, 3], stddev=1), name="weight")
    b = tf.Variable(tf.zeros([3]), name="bias")

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    global_step = tf.train.get_or_create_global_step()

    # Gradient Descent
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

    # Test trained model
    label = tf.argmax(y, 1, name="label")
    prediction = tf.argmax(pred, 1, name="prediction")
    correct_prediction = tf.equal(prediction, label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    init_op = tf.global_variables_initializer()
    return train_op, loss, accuracy, init_op


def linear_regression_layer(x, y):
    # Set model weights
    W = tf.Variable(tf.truncated_normal([13], stddev=1), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")

    global_step = tf.train.get_or_create_global_step()

    # Construct a linear model
    pred = tf.add(tf.multiply(x, W), b)

    # Mean squared error
    # n_samples = x.shape[0]
    # loss = tf.reduce_sum(tf.pow(pred - y_, 2)) / (2 * n_samples)
    loss = tf.reduce_sum(tf.pow(pred - y, 2))
    # Gradient Descent
    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    return train_op, loss, init_op
