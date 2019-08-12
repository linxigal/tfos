#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/15 11:15
:File   : convolution.py
"""

from tensorflow.python.keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose

from tfos.base import BaseLayer, ext_exception


class Conv1DLayer(BaseLayer):
    __doc__ = Conv2D.__doc__

    @ext_exception("Convolution1D Layer")
    def add(self, filters, kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=1,
            activation=None,
            use_bias=True, **kwargs):
        return self._add_layer(Conv1D(filters, kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      use_bias=use_bias,
                                      **kwargs))


class Conv2DLayer(BaseLayer):
    __doc__ = Conv2D.__doc__

    @ext_exception("Convolution2D Layer")
    def add(self, filters, kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True, **kwargs):
        return self._add_layer(Conv2D(filters, kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      activation=activation,
                                      use_bias=use_bias,
                                      **kwargs))


class Conv3DLayer(BaseLayer):
    __doc__ = Conv3D.__doc__

    @ext_exception("Convolution3D Layer")
    def add(self, filters, kernel_size,
            strides=(1, 1, 1),
            padding='valid',
            data_format=None,
            dilation_rate=(1, 1, 1),
            activation=None,
            use_bias=True, **kwargs):
        return self._add_layer(Conv3D(filters, kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      activation=activation,
                                      use_bias=use_bias,
                                      **kwargs))


class Conv2DTransposeLayer(BaseLayer):
    __doc__ = Conv2DTranspose.__doc__

    @ext_exception("Convolution2DTranspose Layer")
    def add(self, filters, kernel_size,
            strides=(1, 1),
            padding='valid',
            output_padding=None,
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True, **kwargs):
        return self._add_layer(Conv2DTranspose(filters, kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               output_padding=output_padding,
                                               data_format=data_format,
                                               dilation_rate=dilation_rate,
                                               activation=activation,
                                               use_bias=use_bias,
                                               **kwargs))


class Conv3DTransposeLayer(BaseLayer):
    __doc__ = Conv3DTranspose.__doc__

    @ext_exception("Convolution3DTranspose Layer")
    def add(self, filters, kernel_size,
            strides=(1, 1, 1),
            padding='valid',
            output_padding=None,
            data_format=None,
            activation=None,
            use_bias=True, **kwargs):
        return self._add_layer(Conv3DTranspose(filters, kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               output_padding=output_padding,
                                               data_format=data_format,
                                               activation=activation,
                                               use_bias=use_bias,
                                               **kwargs))
