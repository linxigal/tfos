#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 12:57
:File       : test_convolution.py
"""


from examples.base import *


class TestConvolution2D(Base):
    def __init__(self, input_model_config_name, filters, kernel_size, strides=(1, 1), activation='relu', input_shape=None):
        super(TestConvolution2D, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('activation', activation)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Convolution2D

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        filters = param.get('filters')
        kernel_size = param.get('kernel_size')
        strides = param.get('strides')
        activation = param.get('activation')
        input_shape = param.get('input_shape')
        model_rdd = inputRDD(input_model_config_name)

        model_config = get_model_config(model_rdd, input_dim=input_shape)
        model = Sequential.from_config(model_config)

        if input_shape:
            model.add(Convolution2D(filters, kernel_size, strides, activation=activation, input_shape=input_shape))
        else:
            model.add(Convolution2D(filters, kernel_size, strides, activation=activation))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    TestConvolution2D('<#zzjzRddName#>', 32, (3, 3), input_shape=(100, 100, 3)).run()
    TestConvolution2D('<#zzjzRddName#>', 64, (3, 3)).run()
    TestConvolution2D('<#zzjzRddName#>', 256, (3, 3)).run()
    print_pretty('<#zzjzRddName#>')
