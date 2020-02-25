# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import json

import tensorflow as tf
from pyspark.sql import Row, DataFrame
from pyspark.sql import SQLContext

from tfos.utils.check import auto_type_checker

INPUTS = "inputs"
OUTPUTS = 'outputs'
METRICS = 'metrics'
PARAMS = 'params'
GRAPH = 'graph'

INPUT_X = 'x'
Y_TRUE = 'y'
PREDICTION = 'y'


class TFBase(object):
    MODEL = 'model'

    def __init__(self):
        self.__inputs = {}
        self.__outputs = {}
        self.__metrics = {}
        self.__params = {}
        self.__graph = None
        self.__INPUTS_CONTAINS_Y = True  # Whether inputs contains 'y'

    def valid_model(self):
        self._check_model()
        self._check_compile()

    def _check_model(self):
        if not self.inputs:
            raise ValueError("TF Model inputs couldn't be empty!")
        if INPUT_X not in self.inputs:
            raise ValueError("TF Model inputs must contains placeholder 'x'!")

    def _check_compile(self):
        if Y_TRUE not in self.inputs:
            raise ValueError("TF Model inputs must contains placeholder 'y'!")
        if not self.outputs:
            raise ValueError("TF Model outputs couldn't be empty!")
        if PREDICTION not in self.outputs:
            raise ValueError("TF Model outputs must contains Tensor 'y'!")

    @classmethod
    @auto_type_checker
    def from_json(cls, data: dict):
        model = cls()
        model.to_self(data)
        return model

    @auto_type_checker
    def connect_model(self, data: dict):
        self.to_self(data)
        return self

    def to_json(self, is_graph=True):
        data = {
            INPUTS: self.__inputs,
            OUTPUTS: self.__outputs,
            METRICS: self.__metrics,
            PARAMS: self.__params
        }
        if is_graph:
            meta_graph = tf.train.export_meta_graph(graph=self.graph, clear_devices=True)
            meta_graph_str = meta_graph.SerializeToString()
            data[GRAPH] = meta_graph_str.decode('latin')
        return data

    @auto_type_checker
    def to_self(self, data: dict):
        self.__inputs.update(data[INPUTS])
        self.__outputs.update(data[OUTPUTS])
        self.__metrics.update(data[METRICS])
        self.__params.update(data[PARAMS])
        if GRAPH in data:
            meta_graph = tf.MetaGraphDef()
            meta_graph.ParseFromString(bytes(data[GRAPH], encoding='latin'))
            tf.train.import_meta_graph(meta_graph)

    @auto_type_checker
    def update_inputs(self, data: dict):
        self.__inputs.update(data)

    @auto_type_checker
    def update_outputs(self, data: dict):
        self.__outputs.update(data)

    @auto_type_checker
    def update_metrics(self, data: dict):
        self.__metrics.update(data)

    @auto_type_checker
    def update_params(self, data: dict):
        self.__params.update(data)

    @property
    def inputs(self):
        inputs = {}
        for key, value in self.__inputs.items():
            if not self.icy and key == Y_TRUE:
                continue
            inputs[key] = self.graph.get_tensor_by_name(value)
        return inputs

    @inputs.setter
    @auto_type_checker
    def inputs(self, value: dict):
        self.__inputs = value

    @property
    def outputs(self):
        outputs = {}
        for key, value in self.__outputs.items():
            outputs[key] = self.graph.get_tensor_by_name(value)
        return outputs

    @outputs.setter
    @auto_type_checker
    def outputs(self, value: dict):
        self.__outputs = value

    @property
    def metrics(self):
        metrics = {}
        for key, value in self.__metrics.items():
            try:
                value = self.graph.get_operation_by_name(value)
            except:
                value = self.graph.get_tensor_by_name(value)
            metrics[key] = value
        return metrics

    @metrics.setter
    @auto_type_checker
    def metrics(self, value: dict):
        self.__metrics = value

    @property
    def params(self):
        return self.__params

    @params.setter
    @auto_type_checker
    def params(self, value: dict):
        self.__params = value

    @property
    def graph(self):
        return self.__graph if self.__graph else tf.get_default_graph()

    @graph.setter
    @auto_type_checker
    def graph(self, value: tf.Graph):
        self.__graph = value

    @property
    def icy(self):
        return self.__INPUTS_CONTAINS_Y

    @icy.setter
    @auto_type_checker
    def icy(self, value: bool):
        self.__INPUTS_CONTAINS_Y = value

    @property
    def output_node_names(self):
        names = [self.__outputs[PREDICTION][:-2]]
        return names


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


class TFMetaClass(type):
    def __new__(mcs, name, bases, attrs):
        if name == "TFLayer":
            bases = (TFBase,)

        if 'build_model' in attrs:
            attrs['build'] = attrs['build_model']
        elif 'compile' in attrs:
            attrs['build'] = attrs['compile']

        new_obj = type.__new__(mcs, name, bases, attrs)

        if name == 'TFModel':
            new_obj.valid = getattr(new_obj, '_check_model')
        elif name == 'TFCompile':
            new_obj.valid = getattr(new_obj, '_check_compile')
        elif name == 'TFComModel':
            new_obj.valid = getattr(new_obj, 'valid_model')
        return new_obj


class TFLayer(metaclass=TFMetaClass):

    def __init__(self):
        super(TFLayer, self).__init__()
        self.__feed_dict = {}

    def add_inputs(self, *args, **kwargs):
        """加入模型的输入张量，placeholder定义的张量

        :param args:
        :param kwargs:
        :return:
        """
        inputs = {}
        for value in args:
            assert isinstance(value, tf.Tensor), "function add_inputs parameter's value must be tf.placeholder!"
            name = value.name
            inputs[name.split(':')[0]] = name
        for key, value in kwargs.items():
            assert isinstance(value, tf.Tensor), "function add_inputs parameter's value must be tf.placeholder!"
            inputs[key] = value.name
        self.update_inputs(inputs)

    def add_params(self, **kwargs):
        """ 传入指定参数，这些参数会在后面模型的编译或者训练时用到

        :param kwargs:
        :return:
        """
        self.update_params(kwargs)

    @property
    def feed_dict(self):
        if not self.__feed_dict:
            for key, value in self.inputs.items():
                self.__feed_dict[value] = self.params.get(key)
        return self.__feed_dict

    def outputs_list(self):
        return list(self.outputs.values())

    @property
    def global_step(self):
        return tf.train.get_or_create_global_step()

    def write_model(self, path, is_graph=True):
        """保存模型

        :param path: 保存模型文件的路径
        :param is_graph: 是否保存模型构建图
        :return:
        """
        with tf.io.gfile.GFile(path, 'w') as f:
            f.write(json.dumps(self.to_json(is_graph), indent=4))

    @classmethod
    def read_model(cls, path):
        with tf.io.gfile.GFile(path, 'r') as f:
            data = json.load(f)
            model = cls.from_json(data)
            return model


class TFModeMiddle(object):

    @auto_type_checker
    def __init__(self, model, sqlc: (type(None), SQLContext) = None, model_rdd: (type(None), DataFrame) = None):
        super(TFModeMiddle, self).__init__()
        self.__model = model
        self.__model_rdd = model_rdd
        self.__sqlc = sqlc

    @property
    @auto_type_checker
    def serialize(self):
        if self.__model_rdd:
            tf.reset_default_graph()
            model_config = json.loads(getattr(self.__model_rdd.first(), self.__model.MODEL))
            self.__model.connect_model(model_config)

        self.__model.build()
        self.__model.valid()

        data = self.__model.to_json()
        return self.__sqlc.createDataFrame([Row(**{self.__model.MODEL: json.dumps(data)})])
