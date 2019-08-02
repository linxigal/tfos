#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 16:57
:File   : __init__.py.py
"""

import json
from pyspark.sql import Row
from .exception import ext_exception
from .config import *

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


class BaseLayer(object):
    def __init__(self, model_rdd, sc=None, sqlc=None):
        self.model_rdd = model_rdd
        self.sc = sc
        self.sqlc = sqlc

    def model2df(self, model, name='model_config'):
        data = {name: json.dumps(model.get_config())}
        return self.sqlc.createDataFrame([Row(**data)])

    def dict2df(self, data, name="model_config"):
        if not isinstance(data, dict):
            raise ValueError("function dict2df: parameter instance must be dict!")
        data = {name: json.dumps(data)}
        return self.sqlc.createDataFrame([Row(**data)])
