#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/6 16:36
:File   : summary.py
"""
import json

from tensorflow.python.keras.models import Sequential

from deep_insight.base import *


class SummaryLayer(Base):
    def __init__(self, name):
        super(SummaryLayer, self).__init__()
        self.p('name', name)

    def run(self):
        params = self.params

        name = params.get('name')
        model_rdd = inputRDD(name)

        model_config = {}
        if model_rdd:
            model_config = json.loads(model_rdd.first().model_config)
        model = Sequential.from_config(model_config)
        model.summary()
        outputRDD('<#zzjzRddName#>_summary', model_rdd)
