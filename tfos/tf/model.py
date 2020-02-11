#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/10 17:22
:File   :graph.py
:content:
  
"""

import json

import tensorflow as tf
from pyspark import RDD
from pyspark.sql import Row, DataFrame
from pyspark.sql import SQLContext
from tfos.utils.check import auto_type_checker

INPUTS = "inputs"
OUTPUTS = 'outputs'
COMPILES = 'compiles'
PARAMS = 'params'
GRAPH = 'graph'


class TFMode(object):
    def __init__(self):
        self.__inputs = []
        self.__outputs = []
        self.__compiles = []
        self.__params = {}
        self.__graph = None

    def serialize(self, sqlc: SQLContext):
        # self.__check_inputs()
        data = self.to_dict()
        return sqlc.createDataFrame([Row(model=json.dumps(data))])

    def to_dict(self):
        graph = self.graph if self.graph else tf.get_default_graph()
        meta_graph = tf.train.export_meta_graph(graph=graph, clear_devices=True)
        meta_graph_str = meta_graph.SerializeToString()
        return {
            INPUTS: self.__inputs,
            OUTPUTS: self.__outputs,
            COMPILES: self.__compiles,
            PARAMS: self.__params,
            GRAPH: meta_graph_str.decode('latin')
        }

    def add_inputs(self, *args):
        for tensor in args:
            name = tensor.name
            if name in self.__inputs:
                raise ValueError("inputs list already exists the tensor name of '{}' !!!".format(name))
            self.__inputs.append(name)

    def __check_inputs(self):
        for name in self.__inputs:
            if name.split(':')[0] not in self.__params:
                raise ValueError("The input tensor type of placeholder '{}' must be assigned!!!")

    def add_outputs(self, *args):
        for tensor in args:
            name = tensor.name
            if name in self.__outputs:
                raise ValueError("outputs list already exists the tensor name of '{}' !!!".format(name))
            self.__outputs.append(name)

    def add_compiles(self, *args):
        count = 1
        for tensor in args:
            print(count, tensor, type(tensor))
            count += 1
            name = tensor.name
            if name in self.__compiles:
                raise ValueError("metrics list already exists the tensor name of '{}' !!!".format(name))
            self.__compiles.append(name)

    def add_params(self, **kwargs):
        self.__params.update(kwargs)

    @property
    def params(self):
        return self.__params

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    @auto_type_checker
    def graph(self, graph: tf.Graph):
        self.__graph = graph

    @auto_type_checker
    def deserialize(self, model_rdd: DataFrame):
        model = model_rdd.first().model
        data = json.loads(model)
        # self.graph = self.graph if self.graph else tf.get_default_graph()
        self.__params = data[PARAMS]
        self.__inputs = data[INPUTS]
        self.__outputs = data[OUTPUTS]

        meta_graph = tf.MetaGraphDef()
        meta_graph.ParseFromString(bytes(data[GRAPH], encoding='latin'))
        tf.train.import_meta_graph(meta_graph)
        return self

    @property
    def feed_dict(self):
        feed_dict = {}
        graph = self.graph if self.graph else tf.get_default_graph()
        for name in self.__inputs:
            feed_dict[graph.get_tensor_by_name(name)] = self.__params[name.split(':')[0]]
        return feed_dict

    @property
    def outputs(self):
        names, tenser_list = [], []
        graph = self.graph if self.graph else tf.get_default_graph()
        for name in self.__outputs:
            names.append(name.split(':')[0])
            tenser_list.append(graph.get_tensor_by_name(name))
        return names, tenser_list

    @property
    def fetches(self):
        names, tenser_list = [], []
        graph = self.graph if self.graph else tf.get_default_graph()
        for name in self.__compiles:
            names.append(name.split(':')[0])
            tenser_list.append(graph.get_tensor_by_name(name))
        return names, tenser_list


@auto_type_checker
def export_inputs(inputs: dict):
    input_dict = {}
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be dict")
    for key, value in inputs.items():
        if not isinstance(key, tf.placeholder):
            raise ValueError("the keys for inputs must be type of tf.placeholder")
        input_dict[key.name] = value
    return input_dict


@auto_type_checker
def export_outputs(outputs: (list, dict)):
    output_dict = {}
    if not isinstance(outputs, (list, dict)):
        raise ValueError("outputs must be list or dict")
    if isinstance(outputs, list):
        for value in outputs:
            if not isinstance(value, tf.Tensor):
                raise ValueError("the values for outputs must be type of tf.Tensor")
            name = value.name
            output_dict[name.split(":")[0]] = name

    else:
        for key, value in outputs.items():
            if not isinstance(value, tf.Tensor):
                raise ValueError("the values for outputs must be type of tf.Tensor")
            output_dict[key] = value.name
    return outputs


@auto_type_checker
def export_model(sqlc: SQLContext, inputs: dict, outputs: (list, dict), graph=None):
    """
    tensorflow模型序列化，并将序列化的结果转化成spark的DataFrame

    :param sqlc:
    :param inputs:
    :param outputs:
    :param graph:
    :return:
    """
    graph = graph if graph else tf.get_default_graph()
    meta_graph = tf.train.export_meta_graph(graph=graph)
    meta_graph_str = meta_graph.SerializeToString()
    data = {
        INPUTS: export_inputs(inputs),
        OUTPUTS: export_outputs(outputs),
        GRAPH: meta_graph_str
    }
    return sqlc.createDataFrame([Row(model=json.dumps(data))])


@auto_type_checker
def import_model(model_rdd: DataFrame, graph=None):
    input_dict = {}
    output_dict = {}
    if not model_rdd:
        raise ValueError("tensorflow model_rdd not exists")
    model = model_rdd.first().model
    data = json.loads(model)
    graph = graph if graph else tf.get_default_graph()

    meta_graph = tf.MetaGraphDef()
    meta_graph.ParseFromString(data[GRAPH])
    tf.train.import_meta_graph(meta_graph)

    for key, value in data[INPUTS].items():
        input_dict[graph.get_tensor_by_name(key)] = value

    for key, value in data[OUTPUTS].items():
        output_dict[key] = graph.get_tensor_by_name(value)

    return input_dict, output_dict


@auto_type_checker
def export_tf_model(sqlc: SQLContext, params: dict, graph=None):
    """
    tensorflow模型序列化，并将序列化的结果转化成spark的DataFrame

    :param sqlc:
    :param params:
    :param graph:
    :return:
    """
    graph = graph if graph else tf.get_default_graph()
    meta_graph = tf.train.export_meta_graph(graph=graph)
    meta_graph_str = meta_graph.SerializeToString()
    data = {
        PARAMS: params,
        GRAPH: meta_graph_str
    }
    return sqlc.createDataFrame([Row(model=json.dumps(data))])


@auto_type_checker
def import_tf_model(model_rdd: DataFrame):
    if not model_rdd:
        raise ValueError("tensorflow model_rdd not exists")
    model = model_rdd.first().model
    data = json.loads(model)
    meta_graph = tf.MetaGraphDef()
    meta_graph.ParseFromString(data[GRAPH])
    tf.train.import_meta_graph(meta_graph)
    return data[PARAMS]
