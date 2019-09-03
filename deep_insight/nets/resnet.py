#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :
:File   :
"""
import unittest

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Model

from deep_insight.base import *


class ResNet50Layer(Base):

    def run(self):
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3))

        # Add final layers
        x = base_model.output
        x = Flatten()(x)
        n_classes = 10
        predictions = Dense(n_classes, activation='softmax', name='fc10')(x)

        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        output_df = model2df(model)
        outputRDD('<#zzjzRddName#>_resnet50', output_df)


class TestResNet50(unittest.TestCase):

    def test_res_net50(self):
        ResNet50Layer()
        SummaryLayer(lrn()).run()
