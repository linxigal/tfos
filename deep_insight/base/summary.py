#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/6 16:36
:File   : summary.py
"""
import json

from tensorflow.python.keras.models import Sequential, Model

from deep_insight.base import *
from tfos.base import ModelType, get_mode_type


class SummaryLayer(Base):
    def __init__(self, name=BRANCH):
        super(SummaryLayer, self).__init__()
        self.p('name', name)

    def run(self):
        params = self.params

        name = params.get('name')
        model_rdd = inputRDD(name)

        if not model_rdd:
            raise ValueError("In Summary model_rdd cannot be empty！")

        model_config = json.loads(model_rdd.first().model_config)
        # model_name = model_config.get('name')
        if get_mode_type(model_rdd) == ModelType.SEQUENCE:
            model = Sequential.from_config(model_config)
        elif get_mode_type(model_rdd) == ModelType.NETWORK:
            model = Model.from_config(model_config)
        else:
            raise ValueError("model type incorrect!!!")

        model.summary()
        outputRDD('<#zzjzRddName#>_Summary', model_rdd)
