#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:28
:File       : test_lstm.py
"""

from examples.base import *


class TestLSTM(Base):
    def __init__(self, input_model_config_name, units):
        super(TestLSTM, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('units', units)

    def run(self):
        param = self.params

        """This layer can only be used as the first layer in a model.
        """
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import LSTM

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        units = param.get("units")

        model_rdd = inputRDD(input_model_config_name)
        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(LSTM(units))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    from examples.test_layer.test_embedding import TestEmbedding

    TestEmbedding('<#zzjzRddName#>', 1024, 256).run()
    TestLSTM('<#zzjzRddName#>', 128).run()
    print_pretty('<#zzjzRddName#>')
