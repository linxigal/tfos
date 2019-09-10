#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:46
:File   : compile.py
"""
import unittest

from deep_insight.base import *
from deep_insight.layers.core import Dense
from deep_insight.layers.input import InputLayer


class Compile(Base):
    """编译层

    模型优化

    参数：
        loss: 损失函数
        optimizer：优化器
        metrics： 评估指标
    """

    def __init__(self, loss, optimizer, metrics=None):
        super(Compile, self).__init__()
        self.p('loss', loss)
        self.p('optimizer', optimizer)
        self.p('metrics', metrics)

    def run(self):
        param = self.params

        from tfos.compile import CompileLayer
        from tfos.choices import LOSSES, OPTIMIZERS, METRICS

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        loss = param.get("loss", LOSSES[0])
        optimizer = param.get('optimizer', OPTIMIZERS[1])
        metrics = param.get('metrics', METRICS[0])

        model_rdd = inputRDD(input_prev_layers)
        output_df = CompileLayer(model_rdd, sc, sqlc).add(loss, optimizer, metrics)
        outputRDD('<#zzjzRddName#>_optimizer', output_df)


class TestOptimizer(unittest.TestCase):

    def test_optimizer(self):
        InputLayer('784').run()
        Dense('512').run()
        Compile('categorical_crossentropy', 'rmsprop', ['accuracy']).run()
        inputRDD(BRANCH).show()


if __name__ == "__main__":
    unittest.main()
