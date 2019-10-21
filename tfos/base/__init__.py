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
    'get_mode_type', 'gmt', 'BaseLayer',
    'ext_exception', 'logger',
    'valid_activations', 'valid_losses', 'valid_metrics', 'valid_optimizers', 'valid_regularizers',
    'ModelType', 'ROUND_NUM'
]

# 保留小数点位数
ROUND_NUM = 6
sequence = 'sequential'
network = 'model'


class ModelType:
    UNKNOWN = -1
    EMPTY = 0
    SEQUENCE = 1
    NETWORK = 2
    COMPILE = 3


def gmt(model_config):
    name = model_config.get('name')
    if name.startswith(sequence):
        return ModelType.SEQUENCE
    elif name.startswith(network):
        return ModelType.NETWORK
    else:
        raise ValueError('model type incorrect!!!')


def get_mode_type(model_rdd):
    names = []
    if model_rdd:
        models = model_rdd if isinstance(model_rdd, list) else [model_rdd]
        for model_rdd in models:
            model = model_rdd.first()
            if 'model_config' in model:
                names.append(gmt(json.loads(model.model_config)))
            else:
                names.append(ModelType.EMPTY)
        names = list(set(names))
        if len(names) > 1:
            raise ValueError("model branch incorrect, {} must be same !!!".format(names))
    else:
        names.append(ModelType.UNKNOWN)
    return names[0]


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
        self.layer_num = 0
        self.layer_name = ""
        # self.__model_type = ModelType.SEQUENCE

    @property
    def model_rdd(self):
        return self.__model_rdd

    @property
    def sc(self):
        return self.__sc

    @property
    def sqlc(self):
        return self.__sqlc

    @property
    def model_name(self):
        names = []
        if not isinstance(self.model_rdd, list):
            models = [self.model_rdd]
        else:
            models = self.model_rdd
        for model_rdd in models:
            if model_rdd:
                model = model_rdd.first()
                if 'model_config' in model:
                    name = json.loads(model.model_config).get('name')
                    names.append(name.split('_')[0])
        names = list(set(names))
        if len(names) > 1:
            raise ValueError("model branch incorrect, {} must be same !!!".format('|'.join(names)))

        return names[0] if names else ''

    def _add_or_create_column(self, name, value):
        if self.model_rdd:
            return self.model_rdd.withColumn(name, value)
        else:
            return self.sqlc.createDataFrame([Row(**{name: value})])

    def model2df(self, model, name='model_config'):
        data = {
            'layer_name': self.layer_name,
            "layer_num": self.layer_num,
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
            self.layer_num += len(model_config.get('layers', []))
        model = Sequential.from_config(model_config)
        model.add(layer)
        self.layer_num += 1
        return self.model2df(model)

    def __add_network_layer(self, layer):
        """网络模型，上一层输入可能有多个模型"""
        model_rdd_list = self.model_rdd
        if not isinstance(model_rdd_list, list):
            model_rdd_list = [model_rdd_list]

        if isinstance(layer, InputLayer):
            output_model = Model(inputs=layer.input, outputs=layer.output)
            self.layer_num += 1
        else:
            if self.model_rdd is None:
                raise ValueError("neural network model first layer must be InputLayer!")
            inputs = []
            outputs = []
            for model_rdd in model_rdd_list:
                model_config = json.loads(model_rdd.first().model_config)
                self.layer_num += len(model_config.get('layers', []))
                input_model = Model.from_config(model_config)
                inputs.extend(input_model.inputs)
                outputs.extend(input_model.outputs)
            outputs = outputs[0] if len(outputs) == 1 else outputs
            output_model = Model(inputs=inputs, outputs=layer(outputs))
            self.layer_num += 1
        return self.model2df(output_model)

    def __add_compile_layer(self, layer):
        return self._add_or_create_column('optimizer', json.dumps(serialize(layer)))

    def _add_layer(self, layer):
        self.layer_name = layer.__class__.__name__

        model_type = get_mode_type(self.model_rdd)

        if model_type == ModelType.NETWORK or isinstance(layer, InputLayer):
            return self.__add_network_layer(layer)
        elif model_type == ModelType.UNKNOWN and isinstance(layer, (Optimizer, OptimizerV2)):
            return self.__add_compile_layer(layer)
        else:
            return self.__add_sequence_layer(layer)

    def add(self):
        raise NotImplementedError
