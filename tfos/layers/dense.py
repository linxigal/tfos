#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 16:39
:File   : dense.py
"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from tfos.base import BaseLayer, get_model_config, ext_exception


class DenseLayer(BaseLayer):

    @ext_exception('Dense Layer')
    def add(self, output_dim, activation=None, input_dim=None):
        if not activation:
            activation = None

        model_config = get_model_config(self.model_rdd, input_dim=input_dim)
        model = Sequential.from_config(model_config)
        model.add(Dense(output_dim, activation=activation, input_dim=input_dim))
        return self.model2df(model)
