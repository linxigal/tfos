#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/15 11:15
:File   : convolution.py
"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, ConvLSTM2D

from tfos.base import BaseLayer, get_model_config, ext_exception


class Convolution2DLayer(BaseLayer):

    @ext_exception("Convolution 2D Layer")
    def add(self, filters, kernel_size):
        pass
