#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 12:53
:File       : optimizer.py
"""

import tensorflow as tf

COST_FUN = ['mse', 'ce']
OPTIMIZER = ['sgd', 'adagrad', 'adam']


def cost_fun(y, pred, cost='mse'):
    if cost == 'mse':
        loss = tf.reduce_sum(tf.pow(pred - y, 2))
    elif cost == 'ce':
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    else:
        raise ValueError('cost function is not exists!')
    return loss


def model_optimizer(loss, opt='adagrad', lr=0.01):
    global_step = tf.train.get_or_create_global_step()
    if opt == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(lr)
    elif opt == 'adagrad':
        train_op = tf.train.AdagradOptimizer(lr)
    else:
        raise ValueError("optimizer parameter is not valid!")
    train_op = train_op.minimize(loss, global_step=global_step)
    return train_op
