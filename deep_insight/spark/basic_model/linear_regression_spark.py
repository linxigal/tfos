#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/27 15:04
:File       : linear_regression.py
"""

from sklearn.datasets import load_boston
from tfos.tfos import TFOSRdd


def load_boston_price():
    data = load_boston()
    x = sc.parallelize(data['data'])
    x = x.map(lambda x: [int(i) for i in x])
    y = sc.parallelize(data['target'])
    y = y.map(lambda x: [int(x)])
    rdd = x.zip(y)
    return rdd


def linear_regression(x, y):
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
    return pred, loss, train_op, global_step


def main(unused_args):
    rdd = load_boston_price()
    tfos = TFOSRdd(sc, linear_regression, 'linear_regression', **FLAGS.flag_values_dict())
    tfos.train(rdd)         # save result to train path
    # tfos.inference(rdd)   # save result to inference path
    # tfos.predict(rdd)     # save result to predict path


if __name__ == "__main__":
    tf.app.run()
