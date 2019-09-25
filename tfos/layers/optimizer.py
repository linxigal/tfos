#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/26 15:33
:File   :optimizer.py
:content:
  
"""

from tensorflow.python.keras.optimizers import *

from tfos.base import BaseLayer, ext_exception


class SGDLayer(BaseLayer):
    __doc__ = SGD.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
        return self._add_layer(SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs))


class RMSpropLayer(BaseLayer):
    __doc__ = RMSprop.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=0.001, rho=0.9, epsilon=None, decay=0., **kwargs):
        return self._add_layer(RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay, **kwargs))


class AdagradLayer(BaseLayer):
    __doc__ = Adagrad.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=0.01, epsilon=None, decay=0., **kwargs):
        return self._add_layer(Adagrad(lr=lr, epsilon=epsilon, decay=decay, **kwargs))


class AdadeltaLayer(BaseLayer):
    __doc__ = Adadelta.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=1.0, rho=0.95, epsilon=None, decay=0., **kwargs):
        return self._add_layer(Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay, **kwargs))


class AdamLayer(BaseLayer):
    __doc__ = Adam.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False, **kwargs):
        return self._add_layer(
            Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad, **kwargs))


class AdamaxLayer(BaseLayer):
    __doc__ = Adamax.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., **kwargs):
        return self._add_layer(Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, **kwargs))


class NadamLayer(BaseLayer):
    __doc__ = Nadam.__doc__

    @ext_exception("SGD Layer")
    def add(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, **kwargs):
        return self._add_layer(
            Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, schedule_decay=schedule_decay, **kwargs))
