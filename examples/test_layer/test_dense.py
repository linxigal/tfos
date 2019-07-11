#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 9:50
:File       : test_dense.py
"""

from examples.base import *


class TestDense(Base):
    def __init__(self, input_model_config_name, output_dim, activation=None, input_dim=None):
        super(TestDense, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('output_dim', output_dim)
        self.p('activation', activation)
        self.p('input_dim', input_dim)

    def run(self):
        param = self.params

        from tfos.layers import DenseLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        output_dim = param.get('output_dim')
        activation = param.get('activation')
        input_dim = param.get('input_dim')

        model_rdd = inputRDD(input_model_config_name)
        dense_layer = DenseLayer(model_rdd, sc, sqlc)
        outputdf = dense_layer.add(output_dim, activation, input_dim)
        outputRDD('<#zzjzRddName#>_dense', outputdf)


if __name__ == "__main__":
    TestDense(lrn(), 512, input_dim=784).run()
    TestDense(lrn(), 256).run()
    TestDense(lrn(), 10).run()
    print_pretty()
