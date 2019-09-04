#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/4 15:15
:File   :input.py
:content:
  
"""

from tensorflow.python.keras.layers import InputLayer as Input

from tfos.base import BaseLayer


class InputLayer(BaseLayer):

    def add(self, input_shape=None,
            batch_size=None,
            dtype=None,
            input_tensor=None,
            sparse=False,
            name=None,
            **kwargs):
        return self._add_layer(Input(
            input_shape=input_shape,
            batch_size=batch_size,
            dtype=dtype,
            input_tensor=input_tensor,
            sparse=sparse,
            name=name,
            **kwargs
        ))
