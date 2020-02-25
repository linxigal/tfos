#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/12 17:03
:File   : core.py
"""

from tensorflow.python.keras.layers import Masking, Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D, \
    Activation, Reshape, Permute, Flatten, RepeatVector, Lambda, Dense, ActivityRegularization

from tfos.base import BaseLayer, ext_exception


class MaskingLayer(BaseLayer):
    __doc__ = Masking.__doc__

    @ext_exception("Masking Layer")
    def add(self, mask_value=0., **kwargs):
        return self._add_layer(Masking(mask_value=mask_value, **kwargs))


class DropoutLayer(BaseLayer):
    __doc__ = Dropout.__doc__

    @ext_exception('Dropout layer')
    def add(self, rate, noise_shape=None, seed=None, **kwargs):
        return self._add_layer(Dropout(rate, noise_shape=noise_shape, seed=seed, **kwargs))


class SpatialDropout1DLayer(BaseLayer):
    __doc__ = SpatialDropout1D.__doc__

    @ext_exception("SpatialDropout1D Layer")
    def add(self, rate, **kwargs):
        return self._add_layer(SpatialDropout1D(rate, **kwargs))


class SpatialDropout2DLayer(BaseLayer):
    __doc__ = SpatialDropout2D.__doc__

    @ext_exception("SpatialDropout2D Layer")
    def add(self, rate, data_format=None, **kwargs):
        return self._add_layer(SpatialDropout2D(rate, data_format=data_format, **kwargs))


class SpatialDropout3DLayer(BaseLayer):
    __doc__ = SpatialDropout3D.__doc__

    @ext_exception("SpatialDropout3D Layer")
    def add(self, rate, data_format=None, **kwargs):
        return self._add_layer(SpatialDropout3D(rate, data_format=data_format, **kwargs))


class ActivationLayer(BaseLayer):
    __doc__ = Activation.__doc__

    @ext_exception("Activation Layer")
    def add(self, activation, **kwargs):
        return self._add_layer(Activation(activation, **kwargs))


class ReshapeLayer(BaseLayer):
    __doc__ = Reshape.__doc__

    @ext_exception("Reshape Layer")
    def add(self, target_shape, **kwargs):
        return self._add_layer(Reshape(target_shape, **kwargs))


class PermuteLayer(BaseLayer):
    __doc__ = Permute.__doc__

    @ext_exception("Permute Layer")
    def add(self, dims, **kwargs):
        return self._add_layer(Permute(dims, **kwargs))


class FlattenLayer(BaseLayer):
    __doc__ = Flatten.__doc__

    @ext_exception("Flatten Layer")
    def add(self, data_format=None, **kwargs):
        return self._add_layer(Flatten(data_format=data_format, **kwargs))


class RepeatVectorLayer(BaseLayer):
    __doc__ = RepeatVector.__doc__

    @ext_exception("RepeatVector Layer")
    def add(self, n, **kwargs):
        return self._add_layer(RepeatVector(n, **kwargs))


class LambdaLayer(BaseLayer):
    __doc__ = Lambda.__doc__

    @ext_exception("Lambda Layer")
    def add(self, function, output_shape=None, mask=None, arguments=None, **kwargs):
        return self._add_layer(Lambda(function, output_shape=output_shape, mask=mask, arguments=arguments, **kwargs))


class DenseLayer(BaseLayer):
    __doc__ = Dense.__doc__

    @ext_exception("Dense Layer")
    def add(self, units,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        return self._add_layer(Dense(units,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     bias_constraint=bias_constraint, **kwargs))


class ActivityRegularizationLayer(BaseLayer):
    __doc__ = ActivityRegularization.__doc__

    @ext_exception("ActivityRegularization Layer")
    def add(self, l1=0., l2=0., **kwargs):
        return self._add_layer(ActivityRegularization(l1=l1, l2=l2, **kwargs))
