#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 12:45
:File       : test_drop.py
"""

from examples.base import *


class TestDrop(Base):
    def __init__(self, input_model_config_name, rate=0.01):
        super(TestDrop, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('rate', rate)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dropout

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        rate = param.get('rate')
        model_rdd = inputRDD(input_model_config_name)

        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(Dropout(rate))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    TestDrop('<#zzjzRddName#>').run()
    TestDrop('<#zzjzRddName#>').run()
    TestDrop('<#zzjzRddName#>').run()
    print_pretty('<#zzjzRddName#>')
