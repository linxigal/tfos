#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/26 15:49
:File   :optimizer.py
:content:
  
"""

import json
import unittest

from deep_insight.base import *


class SGD(Base):
    """随机梯度下降优化器。

    包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov 动量 (NAG) 优化

    参数
        lr: 学习率
            float >= 0. 学习率。
        momentum: 加速度
            float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡。
        decay: 习率衰减值
            float >= 0. 每次参数更新后学习率衰减值。
        nesterov: 牛顿动量
            boolean. 是否使用 Nesterov 动量。
    """

    def __init__(self, lr='0.01', momentum='0.', decay='0.', nesterov='False'):
        super(SGD, self).__init__()
        self.p('lr', lr)
        self.p('momentum', momentum)
        self.p('decay', decay)
        self.p('nesterov', nesterov)

    def run(self):
        param = self.params
        from tfos.layers import SGDLayer
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '0.01')
        momentum = param.get("momentum", '0.')
        decay = param.get("decay", '0.')
        nesterov = param.get("nesterov", BOOLEAN[1])

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if momentum:
            kwargs['momentum'] = float(momentum)
        if decay:
            kwargs['decay'] = float(decay)
        if nesterov:
            kwargs['nesterov'] = convert_bool(nesterov)

        output_df = SGDLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_SGD', output_df)


class RMSprop(Base):
    """RMSProp优化器.

    建议使用优化器的默认参数 （除了学习率 lr，它可以被自由调节）

    这个优化器通常是训练循环神经网络RNN的不错选择。

    参数
        lr: 学习率
            float >= 0. 学习率。
        rho: 梯度移动衰减率
            float >= 0. RMSProp梯度平方的移动均值的衰减率.
        epsilon: 模糊因子
            float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
        decay: 习率衰减值
            float >= 0. 每次参数更新后学习率衰减值。
    """

    def __init__(self, lr='0.001', rho='0.9', epsilon='', decay='0.'):
        super(RMSprop, self).__init__()
        self.p('lr', lr)
        self.p('rho', rho)
        self.p('epsilon', epsilon)
        self.p('decay', decay)

    def run(self):
        param = self.params
        from tfos.layers import RMSpropLayer

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '0.001')
        rho = param.get("rho", '0.9')
        epsilon = param.get("epsilon", '')
        decay = param.get("decay", '0.')

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if rho:
            kwargs['rho'] = float(rho)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if decay:
            kwargs['decay'] = float(decay)

        output_df = RMSpropLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_RMSprop', output_df)


class Adagrad(Base):
    """Adagrad 优化器。

    Adagrad 是一种具有特定参数学习率的优化器，它根据参数在训练期间的更新频率进行自适应调整。参数接收的更新越多，更新越小。

    建议使用优化器的默认参数。

    参数
        lr: 学习率
            float >= 0. 学习率.
        epsilon: 模糊因子
            float >= 0. 若为 None, 默认为 K.epsilon().
        decay: 习率衰减值
            float >= 0. 每次参数更新后学习率衰减值.
    """

    def __init__(self, lr='0.01', epsilon='', decay='0.'):
        super(Adagrad, self).__init__()
        self.p('lr', lr)
        self.p('epsilon', epsilon)
        self.p('decay', decay)

    def run(self):
        param = self.params
        from tfos.layers import AdagradLayer

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '0.01')
        epsilon = param.get("epsilon", '')
        decay = param.get("decay", '0.')

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if decay:
            kwargs['decay'] = float(decay)

        output_df = AdagradLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Adagrad', output_df)


class Adadelta(Base):
    """Adadelta 优化器。

    Adadelta 是 Adagrad 的一个具有更强鲁棒性的的扩展版本，它不是累积所有过去的梯度，而是根据渐变更新的移动窗口调整学习速率。
    这样，即使进行了许多更新，Adadelta 仍在继续学习。 与 Adagrad 相比，在 Adadelta 的原始版本中，您无需设置初始学习率。
    在此版本中，与大多数其他 Keras 优化器一样，可以设置初始学习速率和衰减因子。

    建议使用优化器的默认参数。

    参数
        lr: 学习率
            float >= 0. 学习率，建议保留默认值。
        rho: 梯度移动衰减率
            float >= 0. Adadelta梯度平方移动均值的衰减率。
        epsilon: 模糊因子
            float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
        decay: 学习率衰减值
            float >= 0. 每次参数更新后学习率衰减值。
    """

    def __init__(self, lr='1.0', rho='0.95', epsilon='', decay='0.'):
        super(Adadelta, self).__init__()
        self.p('lr', lr)
        self.p('rho', rho)
        self.p('epsilon', epsilon)
        self.p('decay', decay)

    def run(self):
        param = self.params
        from tfos.layers import AdadeltaLayer

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '1.0')
        rho = param.get("rho", '0.95')
        epsilon = param.get("epsilon", '')
        decay = param.get("decay", '0.')

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if rho:
            kwargs['rho'] = float(rho)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if decay:
            kwargs['decay'] = float(decay)

        output_df = AdadeltaLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Adadelta', output_df)


class Adam(Base):
    """Adam 优化器。

    默认参数遵循原论文中提供的值。

    参数
        lr: 学习率
            float >= 0. 学习率。
        beta_1: beta_1
            float, 0 < beta < 1. 通常接近于 1。
        beta_2: beta_2
            float, 0 < beta < 1. 通常接近于 1。
        epsilon: 模糊因子
            float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
        decay: 学习率衰减值
            float >= 0. 每次参数更新后学习率衰减值。
        amsgrad: AMSGrad变种
            boolean. 是否应用此算法的 AMSGrad 变种，来自论文 "On the Convergence of Adam and Beyond"。
    """

    def __init__(self, lr='0.001', beta_1='0.9', beta_2='0.999', epsilon='', decay='0.', amsgrad='False'):
        super(Adam, self).__init__()
        self.p('lr', lr)
        self.p('beta_1', beta_1)
        self.p('beta_2', beta_2)
        self.p('epsilon', epsilon)
        self.p('decay', decay)
        self.p('amsgrad', amsgrad)

    def run(self):
        param = self.params
        from tfos.layers import AdamLayer
        from tfos.utils import convert_bool
        from tfos.choices import BOOLEAN

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '0.001')
        beta_1 = param.get("beta_1", '0.9')
        beta_2 = param.get("beta_2", '0.999')
        epsilon = param.get("epsilon", '')
        decay = param.get("decay", '0.')
        amsgrad = param.get("amsgrad", BOOLEAN[1])

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if beta_1:
            kwargs['beta_1'] = float(beta_1)
        if beta_2:
            kwargs['beta_2'] = float(beta_2)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if decay:
            kwargs['decay'] = float(decay)
        if amsgrad:
            kwargs['amsgrad'] = convert_bool(amsgrad)

        output_df = AdamLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Adam', output_df)


class Adamax(Base):
    """Adamax 优化器

    Adamax 优化器，来自 Adam 论文的第七小节.

    它是Adam算法基于无穷范数（infinity norm）的变种。 默认参数遵循论文中提供的值。

    参数
        lr: 学习率
            float >= 0. 学习率。
        beta_1: beta_1
            float, 0 < beta < 1. 通常接近于 1。
        beta_2: beta_2
            float, 0 < beta < 1. 通常接近于 1。
        epsilon: 模糊因子
            float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
        decay: 学习率衰减值
            float >= 0. 每次参数更新后学习率衰减值。
    """

    def __init__(self, lr='0.002', beta_1='0.9', beta_2='0.999', epsilon='', decay='0.'):
        super(Adamax, self).__init__()
        self.p('lr', lr)
        self.p('beta_1', beta_1)
        self.p('beta_2', beta_2)
        self.p('epsilon', epsilon)
        self.p('decay', decay)

    def run(self):
        param = self.params
        from tfos.layers import AdamaxLayer

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '0.002')
        beta_1 = param.get("beta_1", '0.9')
        beta_2 = param.get("beta_2", '0.999')
        epsilon = param.get("epsilon", '')
        decay = param.get("decay", '0.')

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if beta_1:
            kwargs['beta_1'] = float(beta_1)
        if beta_2:
            kwargs['beta_2'] = float(beta_2)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if decay:
            kwargs['decay'] = float(decay)

        output_df = AdamaxLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Adamax', output_df)


class Nadam(Base):
    """Nesterov版本Adam优化器。

    正像 Adam 本质上是 RMSProp 与动量 momentum 的结合， Nadam 是采用 Nesterov momentum 版本的 Adam 优化器。

    默认参数遵循论文中提供的值。 建议使用优化器的默认参数。

    参数
        lr:
            float >= 0. 学习率。
        beta_1: beta_1
            float, 0 < beta < 1. 通常接近于 1。
        beta_2: beta_2
            float, 0 < beta < 1. 通常接近于 1。
        epsilon: 模糊因子
            float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
        schedule_decay：规划衰减值
    """

    def __init__(self, lr='0.002', beta_1='0.9', beta_2='0.999', epsilon='', schedule_decay='0.004'):
        super(Nadam, self).__init__()
        self.p('lr', lr)
        self.p('beta_1', beta_1)
        self.p('beta_2', beta_2)
        self.p('epsilon', epsilon)
        self.p('schedule_decay', schedule_decay)

    def run(self):
        param = self.params
        from tfos.layers import NadamLayer

        # param = json.loads('<#zzjzParam#>')
        lr = param.get("lr", '0.002')
        beta_1 = param.get("beta_1", '0.99')
        beta_2 = param.get("beta_2", '0.999')
        epsilon = param.get("epsilon", '')
        schedule_decay = param.get("schedule_decay", '0.004')

        kwargs = {}
        if lr:
            kwargs['lr'] = float(lr)
        if beta_1:
            kwargs['beta_1'] = float(beta_1)
        if beta_2:
            kwargs['beta_2'] = float(beta_2)
        if epsilon:
            kwargs['epsilon'] = float(epsilon)
        if schedule_decay:
            kwargs['schedule_decay'] = float(schedule_decay)

        output_df = NadamLayer(sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Nadam', output_df)


class TestOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self.m = 0

    def tearDown(self) -> None:
        rdd = inputRDD(self.m)
        print(json.dumps(json.loads(rdd.first().optimizer), indent=4))
        reset()

    # @unittest.skip('')
    def test_sgd(self):
        SGD(lr='0.25').b(self.m).run()

    # @unittest.skip('')
    def test_rmsprop(self):
        RMSprop(lr='0.25').b(self.m).run()

    # @unittest.skip('')
    def test_adagrad(self):
        Adagrad(lr='0.25').b(self.m).run()

    # @unittest.skip('')
    def test_adadelta(self):
        Adadelta(lr='0.25').b(self.m).run()

    # @unittest.skip('')
    def test_adam(self):
        Adam(lr='0.25').b(self.m).run()

    # @unittest.skip('')
    def test_adamax(self):
        Adamax(lr='0.25').b(self.m).run()

    # @unittest.skip('')
    def test_nadam(self):
        Nadam(lr='0.25').b(self.m).run()


if __name__ == '__main__':
    unittest.main()
