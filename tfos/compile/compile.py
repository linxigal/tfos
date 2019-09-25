#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/11 15:19
:File   : compile.py
"""
import json

from pyspark.sql.functions import lit
from pyspark.sql.dataframe import DataFrame
from tensorflow.python.keras.optimizers import get, serialize

from tfos.base import *


class CompileLayer(BaseLayer):

    @ext_exception('optimizer layer')
    def add(self, loss, optimizer, metrics):
        self.check_loss(loss)
        optimizer_params = {
            'loss': loss,
            'optimizer': self.valid_optimizer(optimizer),
            'metrics': metrics
        }
        if metrics:
            self.check_metrics(metrics)
            optimizer_params['metrics'] = metrics
        return self._add_or_create_column("compile_config", lit(json.dumps(optimizer_params)))

    @staticmethod
    def check_loss(loss):
        assert loss in valid_losses, 'model loss function incorrect!'

    @staticmethod
    def valid_optimizer(optimizer):
        if optimizer and isinstance(optimizer, dict):
            class_name = optimizer.get('class_name')
            optimizer = get(class_name).from_config(optimizer.get('config', {}))
            optimizer = serialize(optimizer)
        elif isinstance(optimizer, DataFrame):
            optimizer = json.loads(optimizer.first().optimizer)
        return optimizer

    @staticmethod
    def check_metrics(metrics):
        illegal_metrics_set = set(metrics) - set(valid_metrics)
        assert not illegal_metrics_set, "find unknown parameter: {}".format(illegal_metrics_set)
