#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:46
:File   : compile.py
"""
import json
import unittest

from deep_insight.base import *
from deep_insight.layers.core import Dense
from deep_insight.layers.input import InputLayer
from deep_insight.layers.optimizer import SGD


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


class CompileAdvanced(Base):
    """高级编译层

    模型优化

    参数：
        loss: 损失函数
        input_optimizer_layer：优化器
        metrics： 评估指标
    """

    def __init__(self, input_optimizer_layer, loss, metrics=None):
        super(CompileAdvanced, self).__init__()
        self.p('input_optimizer_layer', input_optimizer_layer)
        self.p('loss', loss)
        self.p('metrics', metrics)

    def run(self):
        param = self.params

        from tfos.compile import CompileLayer
        from tfos.choices import LOSSES, METRICS

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        input_optimizer_layer = param.get('input_optimizer_layer')
        loss = param.get("loss", LOSSES[0])
        metrics = param.get('metrics', METRICS[0])

        model_rdd = inputRDD(input_prev_layers)
        optimizer_rdd = inputRDD(input_optimizer_layer)
        output_df = CompileLayer(model_rdd, sc, sqlc).add(loss, optimizer_rdd, metrics)
        outputRDD('<#zzjzRddName#>_optimizer', output_df)


class TestCompile(unittest.TestCase):

    @unittest.skip('')
    def test_compile(self):
        InputLayer('784').run()
        Dense('512').run()
        Compile('categorical_crossentropy', 'rmsprop', ['accuracy']).run()
        inputRDD(BRANCH).show()

    # @unittest.skip('')
    def test_compile_advanced(self):
        # model
        SGD(lr='0.25').b(0).run()

        InputLayer('784').b(1).run()
        Dense('512').run()

        CompileAdvanced(0, 'categorical_crossentropy', ['accuracy']).run()
        rdd = inputRDD(1)
        rdd.show()
        print(json.dumps(json.loads(rdd.first().compile_config), indent=4))
        print(json.dumps(json.loads(rdd.first().model_config), indent=4))


if __name__ == "__main__":
    unittest.main()
