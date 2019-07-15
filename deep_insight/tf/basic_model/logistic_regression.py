#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/27 16:38
:File       : logistic_regression_tf.py
"""

from sklearn.datasets import load_iris
from tfos.tfos import TFOSLocal
from deep_insight import *
from tfos.contrib.layers.regression import logistic_regression
from tfos.contrib.loss import loss_func
from tfos.contrib.optimizer import model_optimizer


def get_data():
    data = load_iris()
    return zip(data['data'], data['target'])


def logistic_regression_v1(x, y):
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
    return pred, loss, train_op, global_step


def build_model(x, y):
    y_pred = logistic_regression(x)
    loss = loss_func(y_pred, y)
    train_op = model_optimizer(loss)


def main(unused_args):
    tfos = TFOSLocal(sc, logistic_regression, 'logistic_regression', **FLAGS.flag_values_dict())
    tfos.train(4)  # save result to train path
    # tfos.inference(rdd)   # save result to inference path
    # tfos.predict(rdd)     # save result to predict path


if __name__ == "__main__":
    tf.app.run()
