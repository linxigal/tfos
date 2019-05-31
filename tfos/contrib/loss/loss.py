#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/31 17:23
:File       : loss.py
"""

import tensorflow as tf

LOSS_FUNC = ['ce', 'mse']


def loss_func(y, y_pred, func='cross_entropy'):
    if func == 'ce':  # 交叉熵
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=1))
    elif func == 'mse':
        loss = tf.reduce_mean(tf.pow((y - y_pred), 2))
    else:
        raise ValueError(f"func is error! choices in {'|'.join(LOSS_FUNC)}")
    return loss
