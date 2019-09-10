#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/8 14:09
:File   : activations.py
"""
from tensorflow.python.keras.layers import LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU

from tfos.base import BaseLayer, ext_exception


class LeakyReLULayer(BaseLayer):
    __doc__ = LeakyReLU.__doc__

    @ext_exception("LeakyReLU Layer")
    def add(self, alpha=0.3, **kwargs):
        return self._add_layer(LeakyReLU(alpha, **kwargs))


class PReLULayer(BaseLayer):
    __doc__ = PReLU.__doc__

    @ext_exception("PReLU Layer")
    def add(self, alpha_initializer='zeros',
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=None, **kwargs):
        return self._add_layer(PReLU(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes, **kwargs))


class ELULayer(BaseLayer):
    __doc__ = ELU.__doc__

    @ext_exception("ELU Layer")
    def add(self, alpha=1.0, **kwargs):
        return self._add_layer(ELU(alpha, **kwargs))


class ThresholdedReLULayer(BaseLayer):
    __doc__ = ThresholdedReLU.__doc__

    @ext_exception("ThresholdedReLU Layer")
    def add(self, theta=1.0, **kwargs):
        return self._add_layer(ThresholdedReLU(theta, **kwargs))


class SoftmaxLayer(BaseLayer):
    __doc__ = Softmax.__doc__

    @ext_exception("Softmax Layer")
    def add(self, axis=-1, **kwargs):
        return self._add_layer(Softmax(axis, **kwargs))


class ReLULayer(BaseLayer):
    __doc__ = ReLU.__doc__

    @ext_exception("ReLU Layer")
    def add(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
        return self._add_layer(ReLU(max_value, negative_slope, threshold, **kwargs))
