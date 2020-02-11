#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/20 16:57
:File   :compile.py
:content:
  
"""

import tensorflow as tf

# from tensorflow.python.framework import ops
from .model import TFMode


class TFCompile(TFMode):
    # def __init__(self):
    #     super(TFCompile, self).__init__()
    #     global_steps = tf.train.get_or_create_global_step()
    #     tf.add_to_collection(ops.GraphKeys.GLOBAL_STEP, global_steps)

    @property
    def global_step(self):
        return tf.train.get_or_create_global_step()
