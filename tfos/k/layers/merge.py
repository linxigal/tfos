#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/3 17:32
:File   :merge.py
:content:
  
"""
from tensorflow.python.keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate, Dot

from tfos.base import BaseLayer, ext_exception


class AddL(BaseLayer):
    __doc__ = Add.__doc__

    @ext_exception("Merge add")
    def add(self, **kwargs):
        return self._add_layer(Add(**kwargs))


class SubtractL(BaseLayer):
    __doc__ = Subtract.__doc__

    def add(self, **kwargs):
        return self._add_layer(Subtract(**kwargs))


class MultiplyL(BaseLayer):
    __doc__ = Multiply.__doc__

    def add(self, **kwargs):
        return self._add_layer(Multiply(**kwargs))


class AverageL(BaseLayer):
    __doc__ = Average.__doc__

    def add(self, **kwargs):
        return self._add_layer(Average(**kwargs))


class MaximumL(BaseLayer):
    __doc__ = Maximum.__doc__

    def add(self, **kwargs):
        return self._add_layer(Maximum(**kwargs))


class MinimumL(BaseLayer):
    __doc__ = Minimum.__doc__

    def add(self, **kwargs):
        return self._add_layer(Minimum(**kwargs))


class ConcatenateL(BaseLayer):
    __doc__ = Concatenate.__doc__

    def add(self, axis=-1, **kwargs):
        return self._add_layer(Concatenate(axis=axis, **kwargs))


class DotL(BaseLayer):
    __doc__ = Dot.__doc__

    def add(self, axes, normalize=False, **kwargs):
        return self._add_layer(Dot(axes=axes, normalize=normalize, **kwargs))
