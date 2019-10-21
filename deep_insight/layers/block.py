#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/21 9:18
:File   :block.py
:content:
  
"""
import unittest
from deep_insight.base import *
from deep_insight.layers.core import Dense, Dropout
from deep_insight.layers.input import InputLayer


class RepeatBegin(Base):
    """复用模块开始
    """

    def __init__(self):
        super(RepeatBegin, self).__init__()

    def run(self):
        param = self.params
        from tfos.layers import RepeatBegin

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")

        model_rdd = inputRDD(input_prev_layers)

        output_df = RepeatBegin(model_rdd).add()
        outputRDD('<#zzjzRddName#>_RepeatBegin', output_df)


class RepeatEnd(Base):
    """结束复用模块

    参数：
        input_start_node: 开始节点
            复用模块开始节点
        repeats: 模块复用次数

    """
    def __init__(self, input_start_node, repeats):
        super(RepeatEnd, self).__init__()
        self.p('input_start_node', input_start_node)
        self.p('repeats', repeats)

    def run(self):
        param = self.params
        from tfos.layers import RepeatEnd

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        input_start_node = param.get("input_start_node")
        repeats = param.get("repeats", '0')

        repeats = int(repeats)

        model_rdd = inputRDD(input_prev_layers)
        start_rdd = inputRDD(input_start_node)
        output_df = RepeatEnd(model_rdd, sqlc=sqlc).add(start_rdd, repeats)
        outputRDD('<#zzjzRddName#>_RepeatEnd', output_df)


class TestBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.b1 = 1
        self.b2 = 2

    def tearDown(self) -> None:
        reset()

    @unittest.skip('')
    def test_repeat_block_sequence(self):
        Dense('256', input_shape='784').b(self.b1).run()
        Dropout('0.5').run()
        RepeatBegin().b(self.b2).run()
        Dense('128', activation='relu').b(self.b1).run()
        Dropout('0.2').run()
        Dense('128', activation='sigmoid').run()
        Dropout('0.8').run()
        RepeatEnd(self.b2, '2').run()
        Dense('10', activation='softmax').run()
        SummaryLayer(self.b1).run()

    @unittest.skip('')
    def test_repeat_block_network(self):
        InputLayer(input_shape='784').b(self.b1).run()
        Dense('256').run()
        Dropout('0.5').run()
        RepeatBegin().b(self.b2).run()
        Dense('128', activation='relu').b(self.b1).run()
        Dropout('0.2').run()
        Dense('128', activation='sigmoid').run()
        Dropout('0.8').run()
        RepeatEnd(self.b2, '2').run()
        Dense('10', activation='softmax').run()
        SummaryLayer(self.b1).run()


if __name__ == '__main__':
    unittest.main()
