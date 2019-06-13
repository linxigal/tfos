#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 11:10
:File       : test_activation.py
"""

from examples.base import *


class TestActivation(Base):
    def __init__(self, inputMutiLayerConfig, fun_name='relu'):
        super(TestActivation, self).__init__()
        self.p('inputMutiLayerConfig', inputMutiLayerConfig)
        self.p('fun_name', fun_name)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Activation

        # param = json.loads('<#zzjzParam#>')
        inputMutiLayerConfig = param.get("inputMutiLayerConfig")
        fun_name = param.get('fun_name')
        model_rdd = inputRDD(inputMutiLayerConfig)

        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(Activation(fun_name))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(inputMutiLayerConfig, outputdf)


if __name__ == "__main__":
    from examples.test_layer.test_dense import TestDense

    TestDense('<#zzjzRddName#>', 512, input_dim=784).run()
    TestActivation('<#zzjzRddName#>').run()
    TestActivation('<#zzjzRddName#>').run()
    TestActivation('<#zzjzRddName#>').run()
    print_pretty('<#zzjzRddName#>')
