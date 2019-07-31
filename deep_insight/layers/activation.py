#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time:      :2019/6/13 11:10
:File       :activation.py
"""

from deep_insight.base import *


class Activation(Base):
    def __init__(self, input_model_config_name, fun_name='relu'):
        super(Activation, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('fun_name', fun_name)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Activation

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        fun_name = param.get('fun_name')
        model_rdd = inputRDD(input_model_config_name)

        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(Activation(fun_name))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    from deep_insight.layers.dense import Dense

    Dense('<#zzjzRddName#>', 512, input_dim=784).run()
    Activation('<#zzjzRddName#>').run()
    Activation('<#zzjzRddName#>').run()
    Activation('<#zzjzRddName#>').run()
    print_pretty('<#zzjzRddName#>')
