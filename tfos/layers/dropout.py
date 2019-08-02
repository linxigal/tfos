#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 17:29
:File   : dropout.py
"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout

from tfos.base import BaseLayer, get_model_config, ext_exception


class DropoutLayer(BaseLayer):

    @ext_exception('dropout layer')
    def add(self, rate, noise_shape=None, seed=None):
        rate = float(rate)
        if not noise_shape:
            noise_shape = None
        if not seed:
            seed = None
        else:
            seed = int(seed)
        model_config = get_model_config(self.model_rdd, False)
        model = Sequential.from_config(model_config)
        model.add(Dropout(rate, noise_shape, seed))
        return self.model2df(model)
