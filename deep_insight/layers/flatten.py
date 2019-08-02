#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:16
:File       : flatten.py
"""

from deep_insight.base import *


class Flatten(Base):
    def __init__(self, input_model_config_name):
        super(Flatten, self).__init__()
        self.p('input_model_config_name', input_model_config_name)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Flatten

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        model_rdd = inputRDD(input_model_config_name)

        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(Flatten())

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    from deep_insight.layers.convolution import Convolution2D
    from deep_insight.layers.max_pooling import MaxPooling2D

    Convolution2D('<#zzjzRddName#>', 32, (3, 3), input_shape=(100, 100, 3)).run()
    MaxPooling2D('<#zzjzRddName#>', (3, 3), (2, 2)).run()
    Flatten('<#zzjzRddName#>').run()
    Flatten('<#zzjzRddName#>').run()
    Flatten('<#zzjzRddName#>').run()
    print_pretty('<#zzjzRddName#>')
