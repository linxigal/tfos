#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/29 17:16
:File       : decorator.py
"""

import tensorflow as tf
from functools import wraps


class batch_export(object):
    def __init__(self, batch_size, capacity, min_after_dequeue):
        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            feature, label = func(*args, **kwargs)
            batch_feature, batch_label = tf.train.shuffle_batch([feature, label],
                                                                batch_size=self.batch_size,
                                                                capacity=self.capacity,
                                                                min_after_dequeue=self.min_after_dequeue)
            return batch_feature, batch_label

        return wrapper
