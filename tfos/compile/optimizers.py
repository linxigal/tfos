#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/25 13:51
:File   :optimizer.py
:content:
  
"""

PARAMS = {}
OPS = {}
OPTIMIZERS = []


class Parameter(object):
    def __init__(self, name, title, desc=None, choices=None):
        self.name = name
        self.title = title
        self.desc = desc or ''
        self.choices = choices or []
        PARAMS[self.name] = vars(self)


p = Parameter
p('lr', '学习率', 'float >= 0. 学习率')
p('momentum', '加速度', 'float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡')
p('decay', '衰减值', 'float >= 0. 每次参数更新后学习率衰减值')
p('nesterov', 'Nesterov动量', 'boolean. 是否使用 Nesterov 动量', ['true', 'false'])
p('rho', '衰减率', 'float >= 0. RMSProp梯度平方的移动均值的衰减率')
p('epsilon', '模糊因子', 'float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()')
p('beta_1', 'beta_1', 'float, 0 < beta < 1. 通常接近于 1')
p('beta_2', 'beta_2', 'float, 0 < beta < 1. 通常接近于 1')
p('amsgrad', '算法变种', 'boolean. 是否应用此算法的 AMSGrad 变种', ['true', 'false'])
p('schedule_decay', '规划衰减值')


def parse(cls):
    params = []
    for key, value in vars(cls()).items():
        ps = PARAMS[key].copy()
        ps['value'] = value
        params.append(ps)
    OPTIMIZERS.append({cls.__name__: params})


@parse
class SGD(object):
    def __init__(self, lr='0.01', momentum='0.', decay='0.', nesterov='False', **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov


@parse
class RMSprop(object):
    def __init__(self, lr='0.001', rho='0.9', epsilon='', decay='0.', **kwargs):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay


@parse
class Adagrad(object):
    def __init__(self, lr='0.01', epsilon='', decay='0.', **kwargs):
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay


@parse
class Adadelta(object):
    def __init__(self, lr='1.0', rho='0.95', epsilon='', decay='0.', **kwargs):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay


@parse
class Adam(object):
    def __init__(self, lr='0.001', beta_1='0.9', beta_2='0.999', epsilon='', decay='0.', amsgrad='False', **kwargs):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.amsgrad = amsgrad


@parse
class Adamax(object):
    def __init__(self, lr='0.002', beta_1='0.9', beta_2='0.999', epsilon='', decay='0.', **kwargs):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay


@parse
class Nadam(object):
    def __init__(self, lr='0.002', beta_1='0.9', beta_2='0.999', epsilon='', schedule_decay=0.004, **kwargs):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
