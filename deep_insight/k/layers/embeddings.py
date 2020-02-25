#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/14 17:05
:File   :embeddings.py
:content:
  
"""

from deep_insight.base import *


class Embedding(Base):
    """Embedding层
    嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]

    Embedding层只能作为模型的第一层

    参数
        input_dim：输入维度
            大或等于0的整数，字典长度，即输入数据最大下标+1
        output_dim：输出维度
            大于0的整数，代表全连接嵌入的维度
        embeddings_initializer: 初始化方法
            嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
        embeddings_regularizer: 正则化
            嵌入矩阵的正则项，为Regularizer对象
        embeddings_constraint: 约束
            嵌入矩阵的约束项，为Constraints对象
        mask_zero：忽略0填充
            布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。
            设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，
            input_dim应设置为|vocabulary| + 1。
        input_length：输入序列长度
            当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，
            否则Dense层的输出维度无法自动推断。

    输入shape
        形如（samples，sequence_length）的2D张量

    输出shape
        形如(samples, sequence_length, output_dim)的3D张量
    """

    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None):
        super(Embedding, self).__init__()
        self.p('input_dim', input_dim)
        self.p('output_dim', output_dim)
        self.p('embeddings_initializer', embeddings_initializer)
        self.p('embeddings_regularizer', embeddings_regularizer)
        self.p('embeddings_constraint', embeddings_constraint)
        self.p('mask_zero', mask_zero)
        self.p('input_length', input_length)

    def run(self):
        param = self.params
        from tfos.k.layers import EmbeddingLayer
        from tfos.utils import convert_bool
        from tfos.choices import BOOLEAN

        # param = json.loads('<#zzjzParam#>')
        input_dim = param.get("input_dim")
        output_dim = param.get("output_dim")
        embeddings_initializer = param.get("embeddings_initializer", '')
        embeddings_regularizer = param.get("embeddings_regularizer", '')
        embeddings_constraint = param.get("embeddings_constraint", '')
        mask_zero = param.get("mask_zero", BOOLEAN[1])
        input_length = param.get("input_length", '')

        kwargs = dict(input_dim=int(input_dim), output_dim=int(output_dim))
        if embeddings_initializer:
            kwargs['embeddings_initializer'] = embeddings_initializer
        if embeddings_regularizer:
            kwargs['embeddings_regularizer'] = embeddings_regularizer
        if embeddings_constraint:
            kwargs['embeddings_constraint'] = embeddings_constraint
        if mask_zero:
            kwargs['mask_zero'] = convert_bool(mask_zero)
        if input_length:
            kwargs['input_length'] = int(input_length)

        output_df = EmbeddingLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Embedding', output_df)


class TestEmbedding(unittest.TestCase):

    def test_embedding(self):
        Embedding('1000', '64', input_length='10').run()
        SummaryLayer().run()


if __name__ == '__main__':
    unittest.main()
