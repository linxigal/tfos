#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/9 9:36
:File   :merge.py
:content:
  
"""

import unittest

from deep_insight.base import *
from deep_insight.layers.core import Dense, Dropout
from deep_insight.layers.input import InputLayer


class Add(Base):
    """张量和
    计算输入张量列表的和。

    它接受一个张量的列表， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）。
    """

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def run(self):
        param = self.params
        from tfos.layers import AddL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)
        output_df = AddL([model_rdd_1, model_rdd_2], sqlc=sqlc).add()
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Subtract(Base):
    """张量差
    计算两个输入张量的差。

    它接受一个长度为 2 的张量列表， 两个张量必须有相同的尺寸，然后返回一个值为 (inputs[0] - inputs[1]) 的张量，
    输出张量和输入张量尺寸相同。
    """

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def run(self):
        param = self.params
        from tfos.layers import SubtractL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = SubtractL([model_rdd_1, model_rdd_2], sqlc=sqlc).add()
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Multiply(Base):
    """张量乘积
    计算输入张量列表的（逐元素间的）乘积。

    它接受一个张量的列表， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）。
    """

    def __init__(self, **kwargs):
        super(Multiply, self).__init__(**kwargs)

    def run(self):
        param = self.params
        from tfos.layers import MultiplyL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = MultiplyL([model_rdd_1, model_rdd_2], sqlc=sqlc).add()
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Average(Base):
    """张量平均值
    计算输入张量列表的平均值。

    它接受一个张量的列表， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）。
    """

    def __init__(self, **kwargs):
        super(Average, self).__init__(**kwargs)

    def run(self):
        param = self.params
        from tfos.layers import AverageL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = AverageL([model_rdd_1, model_rdd_2], sqlc=sqlc).add()
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Maximum(Base):
    """最大张量
    计算输入张量列表的（逐元素间的）最大值。

    它接受一个张量的列表， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）。
    """

    def __init__(self, **kwargs):
        super(Maximum, self).__init__(**kwargs)

    def run(self):
        param = self.params
        from tfos.layers import MaximumL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = MaximumL([model_rdd_1, model_rdd_2], sqlc=sqlc).add()
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Minimum(Base):
    """最小张量
    计算输入张量列表的（逐元素间的）最小值。

    它接受一个张量的列表， 所有的张量必须有相同的输入尺寸， 然后返回一个张量（和输入张量尺寸相同）。
    """

    def __init__(self, **kwargs):
        super(Minimum, self).__init__(**kwargs)

    def run(self):
        param = self.params
        from tfos.layers import MinimumL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = MinimumL([model_rdd_1, model_rdd_2], sqlc=sqlc).add()
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Concatenate(Base):
    """ 张量连接
    连接一个输入张量的列表。

    它接受一个张量的列表， 除了连接轴之外，其他的尺寸都必须相同， 然后返回一个由所有输入张量连接起来的输出张量。

    参数:
        axis: 连接的轴
    """

    def __init__(self, axis='-1', **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.p('axis', axis)

    def run(self):
        param = self.params
        from tfos.layers import ConcatenateL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")
        axis = param.get("axis", '-1')

        axis = int(axis)

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = ConcatenateL([model_rdd_1, model_rdd_2], sqlc=sqlc).add(axis)
        outputRDD('<#zzjzRddName#>_Add', output_df)


class Dot(Base):
    """ 张量点积
    计算两个张量之间样本的点积。

    例如，如果作用于输入尺寸为 (batch_size, n) 的两个张量 a 和 b， 那么输出结果就会是尺寸为 (batch_size, 1) 的一个张量。 在这个张量中，每一个条目 i 是 a[i] 和 b[i] 之间的点积。

    参数:
        axes: 点积轴
            整数或者整数元组， 一个或者几个进行点积的轴。
        normalize: 标准化
            是否在点积之前对即将进行点积的轴进行 L2 标准化。 如果设置成 True，那么输出两个样本之间的余弦相似值。
    """

    def __init__(self, axes, normalize='False', **kwargs):
        super(Dot, self).__init__(**kwargs)
        self.p('axes', axes)
        self.p('normalize', normalize)

    def run(self):
        param = self.params
        from tfos.layers import DotL

        # param = json.loads('<#zzjzParam#>')
        input_branch_1 = param.get("input_branch_1")
        input_branch_2 = param.get("input_branch_2")
        axes = param.get("axes")
        normalize = param.get("normalize", 'false')

        axes = int(axes)
        if normalize.lower() == 'true':
            normalize = True
        else:
            normalize = False

        model_rdd_1 = inputRDD(input_branch_1)
        model_rdd_2 = inputRDD(input_branch_2)

        output_df = DotL([model_rdd_1, model_rdd_2], sqlc=sqlc).add(axes, normalize)
        outputRDD('<#zzjzRddName#>_Add', output_df)


class TestMerge(unittest.TestCase):

    def setUp(self) -> None:
        n1 = BRANCH_1
        n2 = BRANCH_2
        # first branch
        InputLayer('784').b(n1).run()
        Dense('256').run()
        Dropout('0.5').run()
        Dense('64').run()
        Dropout('0.2').run()

        # second branch
        InputLayer('256').b(n2).run()
        Dense('64').run()
        Dropout('0.5').run()

    def tearDown(self) -> None:
        reset()

    # @unittest.skip("")
    def test_add(self):
        Add().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_subtract(self):
        Subtract().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_multiply(self):
        Multiply().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_average(self):
        Average().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_maximum(self):
        Maximum().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_minimum(self):
        Minimum().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_concatenate(self):
        Concatenate().b(3).run()
        SummaryLayer(3).run()

    # @unittest.skip("")
    def test_dot(self):
        Dot('1').b(3).run()
        SummaryLayer(3).run()


if __name__ == '__main__':
    unittest.main()
