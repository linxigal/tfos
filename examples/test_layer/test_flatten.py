#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:16
:File       : test_flatten.py
"""

from examples.base import *


class TestFlatten(Base):
    def __init__(self, inputMutiLayerConfig):
        super(TestFlatten, self).__init__()
        self.p('inputMutiLayerConfig', inputMutiLayerConfig)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Flatten

        # param = json.loads('<#zzjzParam#>')
        inputMutiLayerConfig = param.get("inputMutiLayerConfig")
        model_rdd = inputRDD(inputMutiLayerConfig)

        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(Flatten())

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(inputMutiLayerConfig, outputdf)


if __name__ == "__main__":
    from examples.test_layer.test_convolution import TestConvolution2D
    from examples.test_layer.test_max_pooling import TestMaxPooling2D

    TestConvolution2D('<#zzjzRddName#>', 32, (3, 3), input_shape=(100, 100, 3)).run()
    TestMaxPooling2D('<#zzjzRddName#>', (3, 3), (2, 2)).run()
    TestFlatten('<#zzjzRddName#>').run()
    TestFlatten('<#zzjzRddName#>').run()
    TestFlatten('<#zzjzRddName#>').run()
    print_pretty('<#zzjzRddName#>')
