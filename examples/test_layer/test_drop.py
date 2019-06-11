#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 12:45
:File       : test_drop.py
"""

from examples.base import *


class TestDrop(Base):
    def __init__(self, inputMutiLayerConfig, rate=0.01):
        super(TestDrop, self).__init__()
        self.p('inputMutiLayerConfig', inputMutiLayerConfig)
        self.p('rate', rate)

    def run(self):
        param = self.params

        import json
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dropout
        from pyspark.sql import Row

        # param = json.loads('<#zzjzParam#>')
        inputMutiLayerConfig = param.get("inputMutiLayerConfig")
        rate = param.get('rate')
        model_config = inputRDD(inputMutiLayerConfig)

        if model_config:
            model_config = json.loads(model_config.first()._1)
        else:
            model_config = {}

        model = Sequential.from_config(model_config)

        model.add(Dropout(rate))

        outputdf = sqlc.createDataFrame([Row(json.dumps(model.get_config()))])
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(inputMutiLayerConfig, outputdf)


if __name__ == "__main__":
    TestDrop('<#zzjzRddName#>_drop_1').run()
    TestDrop('<#zzjzRddName#>_drop_2').run()
    TestDrop('<#zzjzRddName#>_drop_3').run()
    print_pretty()
