#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/15 8:33
:File   :normalization.py
:content:
  
"""

from tensorflow.python.keras.layers import BatchNormalization

from tfos.base import BaseLayer, ext_exception


class BatchNormalizationLayer(BaseLayer):
    __doc__ = BatchNormalization.__doc__

    @ext_exception("BatchNormalization Layer")
    def add(self,
            axis=-1,
            momentum=0.99,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            **kwargs):
        return self._add_layer(BatchNormalization(axis=axis,
                                                  momentum=momentum,
                                                  epsilon=epsilon,
                                                  center=center,
                                                  scale=scale,
                                                  beta_initializer=beta_initializer,
                                                  gamma_initializer=gamma_initializer,
                                                  moving_mean_initializer=moving_mean_initializer,
                                                  moving_variance_initializer=moving_variance_initializer,
                                                  beta_regularizer=beta_regularizer,
                                                  gamma_regularizer=gamma_regularizer,
                                                  beta_constraint=beta_constraint,
                                                  gamma_constraint=gamma_constraint, **kwargs))
