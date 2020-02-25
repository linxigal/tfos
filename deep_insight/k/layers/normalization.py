#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/15 8:41
:File   :normalization.py
:content:
  
"""

from deep_insight.base import *
from deep_insight.k.layers.input import InputLayer


class BatchNormalization(Base):
    """
    该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1

    参数
        axis:规范化的轴
            整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=1。
        momentum: 动态均值的动量
        epsilon：防止除0错误
            大于0的小浮点数，用于防止除0错误
        center: beta偏置
            若设为True，将会将beta作为偏置加上去，否则忽略参数beta
        scale: gamma因子
            若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
        beta_initializer：beta权重的初始方法
        gamma_initializer: gamma的初始化方法
        moving_mean_initializer: 动态均值的初始化方法
        moving_variance_initializer: 动态方差的初始化方法
        beta_regularizer: beta正则
            可选的beta正则
        gamma_regularizer: gamma正则
            可选的gamma正则
        beta_constraint: beta约束
            可选的beta约束
        gamma_constraint: gamma约束
            可选的gamma约束
    输入shape
        任意，当使用本层为模型首层时，指定input_shape参数时有意义。

    输出shape
        与输入shape相同
    """

    def __init__(self, axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None):
        super(BatchNormalization, self).__init__()
        self.p('axis', axis)
        self.p('momentum', momentum)
        self.p('epsilon', epsilon)
        self.p('center', center)
        self.p('scale', scale)
        self.p('beta_initializer', beta_initializer)
        self.p('gamma_initializer', gamma_initializer)
        self.p('moving_mean_initializer', moving_mean_initializer)
        self.p('moving_variance_initializer', moving_variance_initializer)
        self.p('beta_regularizer', beta_regularizer)
        self.p('gamma_regularizer', gamma_regularizer)
        self.p('beta_constraint', beta_constraint)
        self.p('gamma_constraint', gamma_constraint)

    def run(self):
        param = self.params
        from tfos.k.layers import BatchNormalizationLayer
        from tfos.utils import convert_bool
        from tfos.choices import BOOLEAN

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        axis = param.get("axis", '-1')
        momentum = param.get("momentum", '0.99')
        epsilon = param.get("epsilon", '0.001')
        center = param.get("center", BOOLEAN[0])
        scale = param.get("scale", BOOLEAN[0])
        beta_initializer = param.get("beta_initializer", 'zeros')
        gamma_initializer = param.get("gamma_initializer", 'ones')
        moving_mean_initializer = param.get("moving_mean_initializer", 'zeros')
        moving_variance_initializer = param.get("moving_variance_initializer", 'ones')
        beta_regularizer = param.get("beta_regularizer", '')
        gamma_regularizer = param.get("gamma_regularizer", '')
        beta_constraint = param.get("beta_constraint", '')
        gamma_constraint = param.get("gamma_constraint", '')

        kwargs = {}
        if axis:
            kwargs['axis'] = int(axis)
        if momentum:
            kwargs['momentum'] = float(momentum)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if center:
            kwargs['center'] = convert_bool(center)
        if scale:
            kwargs['scale'] = convert_bool(scale)
        if beta_initializer:
            kwargs['beta_initializer'] = beta_initializer
        if gamma_initializer:
            kwargs['gamma_initializer'] = gamma_initializer
        if moving_mean_initializer:
            kwargs['moving_mean_initializer'] = moving_mean_initializer
        if moving_variance_initializer:
            kwargs['moving_variance_initializer'] = moving_variance_initializer
        if beta_regularizer:
            kwargs['beta_regularizer'] = beta_regularizer
        if gamma_regularizer:
            kwargs['gamma_regularizer'] = gamma_regularizer
        if beta_constraint:
            kwargs['beta_constraint'] = beta_constraint
        if gamma_constraint:
            kwargs['gamma_constraint'] = gamma_constraint

        model_rdd = inputRDD(input_prev_layers)
        output_df = BatchNormalizationLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_BatchNormalization', output_df)


class TestBatchNormalization(unittest.TestCase):

    def test_batch_normalization(self):
        InputLayer('28,28,1').run()
        BatchNormalization().run()
        SummaryLayer().run()


if __name__ == '__main__':
    unittest.main()
