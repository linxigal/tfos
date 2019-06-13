#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 9:50
:File       : test_dense.py
"""

from examples.base import *


class TestDense(Base):
    def __init__(self, inputMutiLayerConfig, output_dim, activation='relu', input_dim=None):
        super(TestDense, self).__init__()
        self.p('inputMutiLayerConfig', inputMutiLayerConfig)
        self.p('output_dim', output_dim)
        self.p('activation', activation)
        self.p('input_dim', input_dim)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense

        # param = json.loads('<#zzjzParam#>')
        inputMutiLayerConfig = param.get("inputMutiLayerConfig")
        output_dim = param.get('output_dim')
        activation = param.get('activation')
        input_dim = param.get('input_dim')
        model_rdd = inputRDD(inputMutiLayerConfig)

        model_config = get_model_config(model_rdd, input_dim=input_dim)
        model = Sequential.from_config(model_config)

        if input_dim:
            model.add(Dense(output_dim, activation=activation, input_dim=input_dim))
        else:
            model.add(Dense(output_dim, activation=activation))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(inputMutiLayerConfig, outputdf)


if __name__ == "__main__":
    TestDense('<#zzjzRddName#>_dense', 512, input_dim=784).run()
    TestDense('<#zzjzRddName#>_dense', 256).run()
    TestDense('<#zzjzRddName#>_dense', 10).run()
    print_pretty('<#zzjzRddName#>_dense')
