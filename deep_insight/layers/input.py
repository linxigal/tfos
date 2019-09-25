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
    """输入层
    输入层，神经网络图结构中的一个节点。
    
    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create its a placeholder tensor (pass arguments `input_shape`, and
    optionally, `dtype`).
    
    It is generally recommend to use the functional layer API via `Input`,
    (which creates an `InputLayer`) without directly using `InputLayer`.
    
    参数:
     input_shape: 输入空间
        一个尺寸元组（整数），不包含批量大小。 例如，input_shape=(32,) 表明期望的输入是按批次的 32 维向量。
     batch_size: 批次大小
        可选参数，输入批次大小，整数或者None
     dtype: 输入数据类型
        输入所期望的数据类型，字符串表示 (float32, float64, int32...)
     input_tensor: 输入张量
        可选的可封装到 Input 层的现有张量。 如果设定了，那么这个层将不会创建占位符张量。
     sparse: 数据是否稀疏
        一个布尔值，指明需要创建的占位符是否是稀疏的。
     name: 输入层名称
        个可选的层的名称的字符串。 在一个模型中应该是唯一的（不可以重用一个名字两次）。 如未提供，将自动生成。
     """

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
        from tfos.choices import BOOLEAN

        # param = json.loads('<#zzjzParam#>')
        input_shape = param.get("input_shape")
        batch_size = param.get("batch_size", "")
        dtype = param.get("dtype", "")
        sparse = param.get("sparse", BOOLEAN[1])
        name = param.get("name", "")

        # 必填参数
        kwargs = dict(input_shape=tuple([int(i) for i in input_shape.split(',') if i]))
        # 可选参数
        if batch_size:
            kwargs['batch_size'] = int(batch_size)
        if dtype:
            kwargs['dtype'] = dtype
        if sparse:
            kwargs['sparse'] = True if sparse.lower() == 'true' else False
        if name:
            kwargs['name'] = name

        output_df = InputLayer(sqlc=sqlc).add(**kwargs)
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
