#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/23 16:33
:File       : input_layer.py
"""

from tensorflowonspark import TFCluster


def spark_input_layer(rdd, **params):
    params['input_mode'] = TFCluster.InputMode.SPARK
    cluster = TFCluster.run(**params)


def tf_input_layer(**params):
    params['input_mode'] = TFCluster.InputMode.TENSORFLOW
    cluster = TFCluster.run(**params)
