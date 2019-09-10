#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time:      :2019/6/13 11:10
:File       :activation.py
"""
import unittest

from deep_insight.base import *
from deep_insight.layers.core import Dense
from deep_insight.layers.input import InputLayer


class LeakyReLU(Base):
    """LeakyReLU层
    带泄漏的 ReLU。

    当神经元未激活时，它仍允许赋予一个很小的梯度： `f(x) = alpha * x for x < 0, f(x) = x for x >= 0`.

    输入尺寸
        可以是任意的。如果将该层作为模型的第一层， 则需要指定`input_shape`参数 （整数元组，不包含样本数量的维度）。

    输出尺寸
        与输入相同。

    参数:
        alpha: 因子
            float >= 0。负斜率系数。
    """

    def __init__(self, alpha):
        super(LeakyReLU, self).__init__()
        self.p('alpha', alpha)

    def run(self):
        params = self.params

        from tfos.layers import LeakyReLULayer

        input_prev_layers = params.get('input_prev_layers')
        alpha = params.get('alpha', '0.3')

        model_rdd = inputRDD(input_prev_layers)
        output_df = LeakyReLULayer(model_rdd, sc, sqlc).add(float(alpha))
        outputRDD('<#zzjzRddName#>_LeakyReLU', output_df)


class PReLU(Base):
    """PReLU层
    参数化的 ReLU。

    形式： `f(x) = alpha * x for x < 0, f(x) = x for x >= 0`, 其中 alpha 是一个可学习的数组，尺寸与 x 相同。

    输入尺寸
        可以是任意的。如果将这一层作为模型的第一层， 则需要指定 input_shape 参数 （整数元组，不包含样本数量的维度）。

    输出尺寸
        与输入相同。

    参数
        alpha_initializer: 权重初始化
            权重的初始化函数。
        alpha_regularizer: 权重正则化
            权重的正则化方法。
        alpha_constraint: 权重约束
            权重的约束。
        shared_axes: 共享轴
            激活函数共享可学习参数的轴。 例如，如果输入特征图来自输出形状为 (batch, height, width, channels) 的 2D 卷积层，
            而且你希望跨空间共享参数，以便每个滤波器只有一组参数， 可设置 shared_axes=[1, 2]。
    """

    def __init__(self, alpha_initializer):
        super(PReLU, self).__init__()
        self.p('alpha_initializer', alpha_initializer)

    def run(self):
        params = self.params

        from tfos.layers import PReLULayer

        input_prev_layers = params.get('input_prev_layers')
        alpha_initializer = params.get('alpha_initializer', 'zeros')

        model_rdd = inputRDD(input_prev_layers)
        output_df = PReLULayer(model_rdd, sc, sqlc).add(alpha_initializer)
        outputRDD('<#zzjzRddName#>_PReLU', output_df)


class ELU(Base):
    """
    指数线性单元。

    形式： `f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0`.

    输入尺寸
        可以是任意的。如果将这一层作为模型的第一层， 则需要指定 input_shape 参数 （整数元组，不包含样本数量的维度）。

    输出尺寸
        与输入相同。

    参数
        alpha: 因子
            负因子的尺度。
    """

    def __init__(self, alpha):
        super(ELU, self).__init__()
        self.p('alpha', alpha)

    def run(self):
        params = self.params

        from tfos.layers import ELULayer

        input_prev_layers = params.get('input_prev_layers')
        alpha = params.get('alpha', '1.0')

        model_rdd = inputRDD(input_prev_layers)
        output_df = ELULayer(model_rdd, sc, sqlc).add(float(alpha))
        outputRDD('<#zzjzRddName#>_ELU', output_df)


class ThresholdedReLU(Base):
    """ThresholdedReLU层
    带阈值的修正线性单元。

    形式： `f(x) = x for x > theta, f(x) = 0 otherwise`.

    输入尺寸
        可以是任意的。如果将这一层作为模型的第一层， 则需要指定 input_shape 参数 （整数元组，不包含样本数量的维度）。

    输出尺寸
        与输入相同。

    参数
        theta: 阈值
            float >= 0。激活的阈值位。
    """

    def __init__(self, theta):
        super(ThresholdedReLU, self).__init__()
        self.p('theta', theta)

    def run(self):
        params = self.params

        from tfos.layers import ThresholdedReLULayer

        input_prev_layers = params.get('input_prev_layers')
        theta = params.get('theta', '1.0')

        model_rdd = inputRDD(input_prev_layers)
        output_df = ThresholdedReLULayer(model_rdd, sc, sqlc).add(float(theta))
        outputRDD('<#zzjzRddName#>_ThresholdedReLU', output_df)


class Softmax(Base):
    """softmax层
    Softmax 激活函数。

    输入尺寸
        可以是任意的。如果将这一层作为模型的第一层， 则需要指定`input_shape`参数 （整数元组，不包含样本数量的维度）。

    输出尺寸
        与输入相同。

    参数
        axis: 标准化轴
            整数，应用 softmax 标准化的轴。
    """

    def __init__(self, axis):
        super(Softmax, self).__init__()
        self.p('axis', axis)

    def run(self):
        params = self.params

        from tfos.layers import SoftmaxLayer

        input_prev_layers = params.get('input_prev_layers')
        axis = params.get('axis', '-1')

        model_rdd = inputRDD(input_prev_layers)
        output_df = SoftmaxLayer(model_rdd, sc, sqlc).add(int(axis))
        outputRDD('<#zzjzRddName#>_Softmax', output_df)


class ReLU(Base):
    """ReLU层
    ReLU 激活函数。

    使用默认值时，它返回逐个元素的 max(x，0)。

    否则：
        如果 x >= max_value，返回 f(x) = max_value，
        如果 threshold <= x < max_value，返回 f(x) = x,
        否则，返回 f(x) = negative_slope * (x - threshold)。

    输入尺寸
        可以是任意的。如果将这一层作为模型的第一层， 则需要指定 input_shape 参数 （整数元组，不包含样本数量的维度）。

    输出尺寸
        与输入相同。

    参数
        max_value: 最大值
            浮点数，最大的输出值。
        negative_slope: 斜率系数
            float >= 0. 负斜率系数。
        threshold: 阈值
            float。"thresholded activation" 的阈值。
    """

    def __init__(self, max_value, negative_slope, threshold):
        super(ReLU, self).__init__()
        self.p('max_value', max_value)
        self.p('negative_slope', negative_slope)
        self.p('threshold', threshold)

    def run(self):
        params = self.params

        from tfos.layers import ReLULayer

        input_prev_layers = params.get('input_prev_layers')
        max_value = params.get('max_value', '')
        negative_slope = params.get('negative_slope', '0')
        threshold = params.get('threshold', '0')

        if not max_value:
            max_value = None
        else:
            max_value = float(max_value)

        model_rdd = inputRDD(input_prev_layers)
        output_df = ReLULayer(model_rdd, sc, sqlc).add(max_value, float(negative_slope), float(threshold))
        outputRDD('<#zzjzRddName#>_ReLU', output_df)


class TestActivation(unittest.TestCase):
    def tearDown(self) -> None:
        reset()

    # @unittest.skip('')
    def test_leaky_relu(self):
        InputLayer('784').run()
        Dense('512').run()
        LeakyReLU(alpha='0.3').run()
        LeakyReLU(alpha='0.3').run()
        LeakyReLU(alpha='0.3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_prelu(self):
        InputLayer('784').run()
        Dense('512').run()
        PReLU(alpha_initializer='zeros').run()
        PReLU(alpha_initializer='zeros').run()
        PReLU(alpha_initializer='zeros').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_elu(self):
        InputLayer('784').run()
        Dense('512').run()
        ELU(alpha='1.0').run()
        ELU(alpha='1.0').run()
        ELU(alpha='1.0').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_thresholded_relu(self):
        InputLayer('784').run()
        Dense('512').run()
        ThresholdedReLU(theta='1.0').run()
        ThresholdedReLU(theta='1.0').run()
        ThresholdedReLU(theta='1.0').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_softmax(self):
        InputLayer('784').run()
        Dense('512').run()
        Softmax(axis='-1').run()
        Softmax(axis='-1').run()
        Softmax(axis='-1').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_relu(self):
        InputLayer('784').run()
        Dense('512').run()
        ReLU(max_value='', negative_slope='0', threshold='0').run()
        ReLU(max_value='', negative_slope='0', threshold='0').run()
        ReLU(max_value='', negative_slope='0', threshold='0').run()
        SummaryLayer().run()


if __name__ == "__main__":
    unittest.main()
