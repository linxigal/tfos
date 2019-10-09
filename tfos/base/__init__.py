#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 16:57
:File   : __init__.py.py
"""

import json

import numpy as np
import tensorflow as tf
from pyspark.sql import Row
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizers import Optimizer, serialize

from .config import *
from .exception import ext_exception
from .logger import logger

__all__ = [
    'get_model_config', 'BaseLayer',
    'ext_exception', 'logger',
    'valid_activations', 'valid_losses', 'valid_metrics', 'valid_optimizers', 'valid_regularizers',
    'ModelType', 'ROUND_NUM'
]

# 保留小数点位数
ROUND_NUM = 6


class ModelType:
    SEQUENCE = 0
    NETWORK = 1
    COMPILE = 2


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


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tf.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class BaseLayer(object):
    def __init__(self, model_rdd=None, sc=None, sqlc=None):
        self.__model_rdd = model_rdd
        self.__sc = sc
        self.__sqlc = sqlc
        self.__layer_num = 0
        self.__layer_name = ""
        self.__model_type = ModelType.SEQUENCE

    @property
    def model_rdd(self):
        return self.__model_rdd

    @property
    def sc(self):
        return self.__sc

    @property
    def sqlc(self):
        return self.__sqlc

    def _add_or_create_column(self, name, value):
        if self.model_rdd:
            return self.model_rdd.withColumn(name, value)
        else:
            return self.sqlc.createDataFrame([Row(**{name: value})])

    def model2df(self, model, name='model_config'):
        data = {
            'layer_name': self.__layer_name,
            "layer_num": self.__layer_num,
            'model_type': self.__model_type,
            name: json.dumps(model.get_config(), cls=CustomEncoder)
        }
        return self.sqlc.createDataFrame([Row(**data)])

    def dict2df(self, data, name="model_config"):
        if not isinstance(data, dict):
            raise ValueError("function dict2df: parameter instance must be dict!")
        data = {name: json.dumps(data)}
        return self.sqlc.createDataFrame([Row(**data)])

    def __add_sequence_layer(self, layer):
        """序列模型，上一层输入只有一个模型"""

        model_config = {}
        if self.model_rdd:
            model_config = json.loads(self.model_rdd.first().model_config)
            self.__layer_num += len(model_config.get('layers', []))
        model = Sequential.from_config(model_config)
        model.add(layer)
        self.__layer_num += 1
        return self.model2df(model)

    def __add_network_layer(self, layer):
        """网络模型，上一层输入可能有多个模型"""
        model_rdd_list = self.model_rdd
        if not isinstance(model_rdd_list, list):
            model_rdd_list = [model_rdd_list]

        if isinstance(layer, InputLayer):
            output_model = Model(inputs=layer.input, outputs=layer.output)
            self.__layer_num += 1
        else:
            if self.model_rdd is None:
                raise ValueError("neural network model first layer must be InputLayer!")
            inputs = []
            outputs = []
            for model_rdd in model_rdd_list:
                model_config = json.loads(model_rdd.first().model_config)
                self.__layer_num += len(model_config.get('layers', []))
                input_model = Model.from_config(model_config)
                inputs.extend(input_model.inputs)
                outputs.extend(input_model.outputs)
            outputs = outputs[0] if len(outputs) == 1 else outputs
            output_model = Model(inputs=inputs, outputs=layer(outputs))
            self.__layer_num += 1
        return self.model2df(output_model)

    def __add_compile_layer(self, layer):
        return self._add_or_create_column('optimizer', json.dumps(serialize(layer)))

    def _add_layer(self, layer):
        self.__layer_name = layer.__class__.__name__
        if self.model_rdd:
            if isinstance(self.model_rdd, list):
                self.__model_type = self.model_rdd[0].first().model_type
            else:
                self.__model_type = self.model_rdd.first().model_type
        else:
            if isinstance(layer, InputLayer):
                self.__model_type = ModelType.NETWORK
            elif isinstance(layer, (Optimizer, OptimizerV2)):
                self.__model_type = ModelType.COMPILE
            else:
                self.__model_type = ModelType.SEQUENCE

        if self.__model_type == ModelType.NETWORK:
            return self.__add_network_layer(layer)
        elif self.__model_type == ModelType.SEQUENCE:
            return self.__add_sequence_layer(layer)
        elif self.__model_type == ModelType.COMPILE:
            return self.__add_compile_layer(layer)
        else:
            raise ValueError("unknown model type!!!")

    def add(self):
        raise NotImplementedError
