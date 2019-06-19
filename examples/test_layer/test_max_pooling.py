#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:01
:File       : test_max_pooling.py
"""

from examples.base import *


class TestMaxPooling2D(Base):
    def __init__(self, input_model_config_name, pool_size, strides, padding='valid'):
        super(TestMaxPooling2D, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import MaxPooling2D

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        pool_size = param.get('pool_size')
        strides = param.get('strides')
        padding = param.get('padding')
        model_rdd = inputRDD(input_model_config_name)

        model_config = get_model_config(model_rdd, False)
        model = Sequential.from_config(model_config)

        model.add(MaxPooling2D(pool_size, strides, padding))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    from examples.test_layer.test_convolution import TestConvolution2D

    TestConvolution2D('<#zzjzRddName#>', 32, (3, 3), input_shape=(100, 100, 3)).run()
    TestMaxPooling2D('<#zzjzRddName#>', (3, 3), (2, 2)).run()
    TestMaxPooling2D('<#zzjzRddName#>', (3, 3), (2, 2)).run()
    TestMaxPooling2D('<#zzjzRddName#>', (3, 3), (2, 2)).run()
    print_pretty('<#zzjzRddName#>')
