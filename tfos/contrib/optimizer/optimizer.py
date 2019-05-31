#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 12:53
:File       : optimizer.py
"""

import tensorflow as tf

OPTIMIZER = ['sgd', 'adagrad', 'adam']


def model_optimizer(loss, opt='adagrad', lr=0.01):
    global_step = tf.train.get_or_create_global_step()
    if opt == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(lr)
    elif opt == 'adagrad':
        train_op = tf.train.AdagradOptimizer(lr)
    else:
        raise ValueError(f"optimizer parameter is not valid! choices {'|'.join(OPTIMIZER)}")
    train_op = train_op.minimize(loss, global_step=global_step)
    return train_op
