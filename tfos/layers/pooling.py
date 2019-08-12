#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/12 15:17
:File   : pooling.py
"""

from tensorflow.python.keras.layers import MaxPool1D, AvgPool1D, MaxPool2D, AvgPool2D, MaxPool3D, AvgPool3D

from tfos.base import BaseLayer, ext_exception


class MaxPool1DLayer(BaseLayer):
    __doc__ = MaxPool1D.__doc__

    @ext_exception("MaxPooling1D Layer")
    def add(self, pool_size=2, strides=None, padding='valid', data_format='channels_last', **kwargs):
        return self._add_layer(MaxPool1D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format, **kwargs))


class AvgPool1DLayer(BaseLayer):
    __doc__ = AvgPool1D.__doc__

    @ext_exception("AveragePooling1D Layer")
    def add(self, pool_size=2, strides=None, padding='valid', data_format='channels_last', **kwargs):
        return self._add_layer(AvgPool1D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format, **kwargs))


class MaxPool2DLayer(BaseLayer):
    __doc__ = MaxPool2D.__doc__

    @ext_exception("MaxPooling2D Layer")
    def add(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        return self._add_layer(MaxPool2D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format, **kwargs))


class AvgPool2DLayer(BaseLayer):
    __doc__ = AvgPool2D.__doc__

    @ext_exception("AveragePooling2D Layer")
    def add(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        return self._add_layer(AvgPool2D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format, **kwargs))


class MaxPool3DLayer(BaseLayer):
    __doc__ = MaxPool3D.__doc__

    @ext_exception("MaxPooling3D Layer")
    def add(self, pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        return self._add_layer(MaxPool3D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format, **kwargs))


class AvgPool3DLayer(BaseLayer):
    __doc__ = AvgPool3D.__doc__

    @ext_exception("AveragePooling3D Layer")
    def add(self, pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        return self._add_layer(AvgPool3D(pool_size=pool_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format, **kwargs))
