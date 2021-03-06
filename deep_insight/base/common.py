#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 14:22
:File       : common.py
"""

import json
from pyspark.sql import Row
from deep_insight.base import sqlc


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


def model2df(model, name='model_config'):
    data = {name: json.dumps(model.get_config())}
    return sqlc.createDataFrame([Row(**data)])


def dict2df(data, name="model_config"):
    if not isinstance(data, dict):
        raise ValueError("function dict2df: parameter instance must be dict!")
    data = {name: json.dumps(data)}
    return sqlc.createDataFrame([Row(**data)])
