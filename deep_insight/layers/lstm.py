#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:28
:File       : lstm.py
"""

from deep_insight.base import *


class LSTM(Base):
    def __init__(self, input_model_config_name, units):
        super(LSTM, self).__init__()
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
    from deep_insight.layers.embedding import Embedding

    Embedding('<#zzjzRddName#>', 1024, 256).run()
    LSTM('<#zzjzRddName#>', 128).run()
    print_pretty('<#zzjzRddName#>')
