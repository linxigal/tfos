#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/21 8:57
:File   :block.py
:content:
  
"""

import json

from tensorflow.python.keras.layers import deserialize
from tensorflow.python.keras.models import Model, Sequential

from tfos.base import BaseLayer, ext_exception
from tfos.config import MODEL_CONFIG


class RepeatBegin(BaseLayer):
    @ext_exception("RepeatBegin Layer")
    def add(self):
        if MODEL_CONFIG not in self.model_rdd.first():
            raise ValueError('repeat units start node not exists model_config!')
        return self.model_rdd


class RepeatEnd(BaseLayer):
    def __init__(self, *args, **kwargs):
        super(RepeatEnd, self).__init__(*args, **kwargs)
        self.model = None

    @ext_exception("RepeatEnd Layer")
    def add(self, start_rdd, repeats=0):
        if MODEL_CONFIG not in self.model_rdd.first():
            raise ValueError('repeat units end node not exists model_config!')
        model_config = json.loads(getattr(self.model_rdd.first(), MODEL_CONFIG))

        start_config = json.loads(getattr(start_rdd.first(), MODEL_CONFIG))
        marker_layer = start_config['layers'][-1]

        for index, layer in enumerate(model_config['layers']):
            if marker_layer == layer:
                layers = model_config['layers'][index + 1:]
                if 'inbound_nodes' in layer:
                    self.model = Model.from_config(model_config)
                    self.repeat_networks(layers, repeats)
                elif 'name' in layer['config']:
                    self.model = Sequential.from_config(model_config)
                    self.repeat_sequence(layers, repeats)
                else:
                    raise ValueError("In RepeatBlock node, model type incorrect!")
        return self.model2df(self.model)

    def repeat_sequence(self, layers, repeats):
        for i in range(repeats):
            for lv in layers:
                layer = deserialize(lv)
                layer._name = layer.name + '_repeat_{}'.format(i + 1)
                self.layer_name = layer.name
                self.layer_num += 1
                self.model.add(layer)

    def repeat_networks(self, layers, repeats):
        for i in range(repeats):
            for lv in layers:
                layer = deserialize(lv)
                layer._name = layer.name + '_repeat_{}'.format(i + 1)
                self.layer_name = layer.name
                self.layer_num += 1
                output = layer(self.model.output)
                self.model = Model(inputs=self.model.inputs, outputs=output)
