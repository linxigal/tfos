#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/10 17:22
:File   :graph.py
:content:
  
"""

import tensorflow as tf

from .base import TFLayer


class TFMode(TFLayer):

    def build_model(self):
        raise NotImplementedError

    def add_outputs(self, *args, **kwargs):
        """模型的输出值

        :param args:
        :param kwargs:
        :return:
        """
        outputs = {}
        for value in args:
            assert isinstance(value, tf.Tensor), "function add_outputs parameter's value must be tf.Tensor"
            name = value.name
            outputs[name.split(':')[0]] = name
        for key, value in kwargs.items():
            assert isinstance(value, tf.Tensor), "function add_outputs parameter's value must be tf.Tensor"
            outputs[key] = value.name
        self.update_outputs(outputs)


class TFCompile(TFLayer):

    def compile(self):
        raise NotImplementedError

    def add_metrics(self, *args, **kwargs):
        """加入模型的评估指标、优化操作等，例如损失值，正确率等张量或者操作

        :param args:
        :param kwargs:
        :return:
        """
        metrics = {}
        for value in args:
            assert isinstance(value, (tf.Operation, tf.Tensor)), \
                "function add_metrics parameter's value must be tf.Operation"
            name = value.name
            metrics[name.split(':')[0]] = name
        for key, value in kwargs.items():
            assert isinstance(value, (tf.Operation, tf.Tensor)), \
                "function add_metrics parameter's value must be tf.Operation"
            metrics[key] = value.name
        self.update_metrics(metrics)

    @property
    def fetches(self):
        """ 获取模型输出值或者评估值， 来优化训练模型

        :return:
        """
        return self.metrics


class TFComModel(TFMode, TFCompile):
    """
    基于TensorFlow的复合模型，即使用一个算子构建模型的和模型的编译
    """

    def build_model(self):
        raise NotImplementedError

    def compile(self):
        pass
