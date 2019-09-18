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
from tensorflow.python.keras.models import Sequential, Model

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


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tf.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class BaseLayer(object):
    def __init__(self, model_rdd, sc=None, sqlc=None):
        if model_rdd and not isinstance(model_rdd, list):
            self.__model_rdd = [model_rdd]
        else:
            self.__model_rdd = model_rdd

        self.__sc = sc
        self.__sqlc = sqlc
        self.__is_sequence = True
        self.__layer_num = 0
        self.__layer_name = ""

    def _add_column(self, name, value):
        if self.__model_rdd:
            if len(self.__model_rdd) == 1:
                return self.__model_rdd[0].withColumn(name, value)
            else:
                raise ValueError("model_rdd length must be one when rdd add column !")
        else:
            raise ValueError("model_rdd cannot empty!")

    def model2df(self, model, name='model_config'):
        data = {
            'layer_name': self.__layer_name,
            "is_sequence": self.__is_sequence,
            "layer_num": self.__layer_num,
            name: json.dumps(model.get_config(), cls=CustomEncoder)
        }
        return self.__sqlc.createDataFrame([Row(**data)])

    def dict2df(self, data, name="model_config"):
        if not isinstance(data, dict):
            raise ValueError("function dict2df: parameter instance must be dict!")
        data = {name: json.dumps(data)}
        return self.__sqlc.createDataFrame([Row(**data)])

    def __add_sequence_layer(self, layer):
        """序列模型，上一层输入只有一个模型"""

        model_rdd = self.__model_rdd[0] if self.__model_rdd else self.__model_rdd
        model_config = {}
        if self.__model_rdd:
            model_config = json.loads(model_rdd.first().model_config)
            self.__layer_num += len(model_config.get('layers', []))
        model = Sequential.from_config(model_config)
        model.add(layer)
        self.__layer_num += 1
        return self.model2df(model)

    def __add_network_layer(self, layer):
        """网络模型，上一层输入可能有多个模型"""

        if self.__layer_name == "InputLayer":
            output_model = Model(inputs=layer.input, outputs=layer.output)
            self.__layer_num += 1
        else:
            if self.__model_rdd is None:
                raise ValueError("neural network model first layer must be InputLayer!")
            inputs = []
            outputs = []
            for model_rdd in self.__model_rdd:
                model_config = json.loads(model_rdd.first().model_config)
                self.__layer_num += len(model_config.get('layers', []))
                input_model = Model.from_config(model_config)
                inputs.extend(input_model.inputs)
                outputs.extend(input_model.outputs)
            outputs = outputs[0] if len(outputs) == 1 else outputs
            output_model = Model(inputs=inputs, outputs=layer(outputs))
            self.__layer_num += 1
        return self.model2df(output_model)

    def _add_layer(self, layer):
        self.__layer_name = layer.__class__.__name__
        if self.__layer_name == "InputLayer":
            self.__is_sequence = False
            if self.__model_rdd:
                raise ValueError("InputLayer must be first Layer, previous layer must be empty!")
        elif self.__model_rdd:
            self.__is_sequence = self.__model_rdd[0].first().is_sequence

        if self.__is_sequence:
            return self.__add_sequence_layer(layer)
        else:
            return self.__add_network_layer(layer)

    def add(self):
        raise NotImplementedError
