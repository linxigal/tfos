#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 16:57
:File   : __init__.py.py
"""

import json

import numpy as np
from pyspark.sql import Row
from tensorflow.python.keras.models import Sequential

from .config import *
from .exception import ext_exception

__all__ = [
    'get_model_config', 'BaseLayer',
    'ext_exception',
    'valid_activations', 'valid_losses', 'valid_metrics', 'valid_optimizers', 'valid_regularizers',
]


def get_model_config(model_rdd, can_first_layer=True, input_dim=None):
    model_config = {}
    if model_rdd:
        model_config = json.loads(model_rdd.first().model_config)
    else:
        if can_first_layer:
            if not input_dim:
                raise ValueError("current node is first layer, the parameter input_dim must be positive!")
        else:
            raise ValueError('current node cannot be first layer!')
    return model_config


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BaseLayer(object):
    def __init__(self, model_rdd, sc=None, sqlc=None):
        self.model_rdd = model_rdd
        self.sc = sc
        self.sqlc = sqlc

    def model2df(self, model, name='model_config'):
        data = {name: json.dumps(model.get_config(), cls=NumpyEncoder)}
        return self.sqlc.createDataFrame([Row(**data)])

    def dict2df(self, data, name="model_config"):
        if not isinstance(data, dict):
            raise ValueError("function dict2df: parameter instance must be dict!")
        data = {name: json.dumps(data)}
        return self.sqlc.createDataFrame([Row(**data)])

    def _add_layer(self, layer):
        model_config = {}
        if self.model_rdd:
            model_config = json.loads(self.model_rdd.first().model_config)
        model = Sequential.from_config(model_config)
        model.add(layer)
        return self.model2df(model)

    def summary(self):
        model_config = {}
        if self.model_rdd:
            model_config = json.loads(self.model_rdd.first().model_config)
        model = Sequential.from_config(model_config)
        model.summary()

    def add(self):
        raise NotImplementedError
