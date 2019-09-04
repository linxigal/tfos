#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/4 16:49
:File   :input.py
:content:
  
"""

import unittest

from deep_insight.base import *


class InputLayer(Base):

    def __init__(self, input_shape=None,
                 batch_size=None,
                 dtype=None,
                 input_tensor=None,
                 sparse=False,
                 name=None):
        super(InputLayer, self).__init__()
        self.p('input_shape', input_shape)
        self.p('batch_size', batch_size)
        self.p('dtype', dtype)
        self.p('input_tensor', input_tensor)
        self.p('sparse', sparse)
        self.p('name', name)

    def run(self):
        param = self.params
        from tfos.layers import InputLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        input_shape = param.get("input_shape")
        batch_size = param.get("batch_size", "")
        # dtype = param.get("dtype", "")
        # input_tensor = param.get("input_tensor", "")
        # sparse = param.get("sparse", "")
        # name = param.get("name", "")

        # 必填参数
        kwargs = dict(input_shape=tuple([int(i) for i in input_shape.split(',') if i]))
        # 可选参数
        if batch_size:
            kwargs['batch_size'] = int(batch_size)

        model_rdd = inputRDD(input_prev_layers)
        output_df = InputLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        # output_df.show()
        outputRDD('<#zzjzRddName#>_Input', output_df)


class TestInput(unittest.TestCase):
    # @unittest.skip("")
    def test_input(self):
        InputLayer('784').b(1).run()
        InputLayer('256').b(2).run()
        InputLayer('64').b(3).run()
        SummaryLayer(2).run()


if __name__ == '__main__':
    unittest.main()
