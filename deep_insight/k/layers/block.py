#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/21 9:18
:File   :block.py
:content:
  
"""
from deep_insight.base import *
from deep_insight.k.layers.core import Dense, Dropout
from deep_insight.k.layers.input import InputLayer
from deep_insight.k.layers.merge import Add


class RepeatBegin(Base):
    """复用模块开始
    """

    def __init__(self):
        super(RepeatBegin, self).__init__()

    def run(self):
        param = self.params
        from tfos.k.layers import RepeatBegin

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
        from tfos.k.layers import RepeatEnd

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        input_start_node = param.get("input_start_node")
        repeats = param.get("repeats", '0')

        repeats = int(repeats)

        model_rdd = inputRDD(input_prev_layers)
        start_rdd = inputRDD(input_start_node)
        output_df = RepeatEnd(model_rdd, sqlc=sqlc).add(start_rdd, repeats)
        outputRDD('<#zzjzRddName#>_RepeatEnd', output_df)


class RepeatBranch(Base):
    """复用分支层

    连续多次复用某个分支模型

    参数：
        input_branch_node: 分支尾部节点
            复用模块开始节点
        repeats: 模块复用次数

    """

    def __init__(self, input_branch_node, repeats):
        super(RepeatBranch, self).__init__()
        self.p('input_branch_node', input_branch_node)
        self.p('repeats', repeats)

    def run(self):
        param = self.params
        from tfos.k.layers import RepeatBranch

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        input_branch_node = param.get("input_branch_node")
        repeats = param.get("repeats", '0')

        repeats = int(repeats)

        model_rdd = inputRDD(input_prev_layers)
        branch_rdd = inputRDD(input_branch_node)
        output_df = RepeatBranch(model_rdd, sqlc=sqlc).add(branch_rdd, repeats)
        outputRDD('<#zzjzRddName#>_RepeatEnd', output_df)


class TestBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.b1 = 1
        self.b2 = 2
        self.b3 = 3
        self.b4 = 4

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

    @unittest.skip('')
    def test_repeat_block_network_branch(self):
        # TODO 暂不兼容复用分支快包含多分支类型
        InputLayer(input_shape='784').b(self.b1).run()
        Dense('256').run()
        Dropout('0.5').run()
        RepeatBegin().b(self.b2).run()
        Dense('128', activation='relu').b(self.b3).run()
        Dropout('0.2').run()
        Dense('128', activation='sigmoid').b(self.b4).run()
        Dropout('0.8').run()
        Add(input_branch_1=self.b3, input_branch_2=self.b4).b(self.b1).run()
        RepeatEnd(self.b2, '2').run()
        Dense('10', activation='softmax').run()
        SummaryLayer(self.b1).run()

    @unittest.skip('')
    def test_repeat_branch(self):
        # branch
        Dense('256').b(self.b1).run()
        Dropout('0.5').run()
        Dense('64').run()

        # main Branch
        InputLayer(input_shape='784').b(self.b2).run()
        Dense('256').run()
        Dropout('0.5').run()
        Dense('128').run()

        RepeatBranch(self.b1, '2').run()
        Dense('10', activation='softmax').run()
        SummaryLayer(self.b2).run()

    @unittest.skip('')
    def test_repeat_branch_multi_branch(self):
        # TODO 暂不兼容复用分支快包含多分支类型
        # branch
        Dense('256').b(self.b1).run()
        Dropout('0.5').run()
        Dense('64').b(self.b2).run()
        Dropout('0.5').run()
        Dense('64').b(self.b1).run()
        Dropout('0.5').run()
        Add(input_branch_1=self.b1, input_branch_2=self.b2).run()

        # main Branch
        InputLayer(input_shape='784').b(self.b2).run()
        Dense('256').run()
        Dropout('0.5').run()
        Dense('128').run()

        RepeatBranch(self.b1, '2').run()
        Dense('10', activation='softmax').run()
        SummaryLayer(self.b2).run()


if __name__ == '__main__':
    unittest.main()
