#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
:Author     :weijinlong
:Time: 2019/5/22 17:18
:File       : logicstic_regression.py
"""

from sklearn.datasets import load_iris
from tfos.tfos import TFOSRdd


def load_iris_data():
    data = load_iris()
    x = sc.parallelize(data['data'])
    x = x.map(lambda x: [int(i) for i in x])
    y = sc.parallelize(data['target'])
    y = y.map(lambda x: [int(x)])
    rdd = x.zip(y)
    return rdd


def logistic_regression(x, y):
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


def main(unused_args):
    rdd = load_iris_data()
    tfos = TFOSRdd(sc, logistic_regression, 'logistic_regression', **FLAGS.flag_values_dict())
    tfos.train(rdd)         # save result to train path
    # tfos.inference(rdd)   # save result to inference path
    # tfos.predict(rdd)     # save result to predict path


if __name__ == "__main__":
    tf.app.run()
