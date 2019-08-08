#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time:      :2019/6/13 11:10
:File       :activation.py
"""
import unittest

from deep_insight.base import *
from deep_insight.layers.dense import Dense


class Activation(Base):
    def __init__(self, input_prev_layers, activation):
        super(Activation, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('activation', activation)

    def run(self):
        params = self.params

        from tfos.layers import ActivationLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = params.get("input_prev_layers")
        activation = params.get('activation')
        model_rdd = inputRDD(input_prev_layers)
        output_df = ActivationLayer(model_rdd, sc, sqlc).add(activation)
        outputRDD('<#zzjzRddName#>_activation', output_df)


class LeakyReLU(Base):
    def __init__(self, input_prev_layers, alpha):
        super(LeakyReLU, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('alpha', alpha)

    def run(self):
        params = self.params

        from tfos.layers import LeakyReLULayer

        input_prev_layers = params.get('input_prev_layers')
        alpha = params.get('alpha', '0.3')

        model_rdd = inputRDD(input_prev_layers)
        output_df = LeakyReLULayer(model_rdd, sc, sqlc).add(float(alpha))
        outputRDD('<#zzjzRddName#>_activation', output_df)


class PReLU(Base):
    def __init__(self, input_prev_layers, alpha_initializer):
        super(PReLU, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('alpha_initializer', alpha_initializer)

    def run(self):
        params = self.params

        from tfos.layers import PReLULayer

        input_prev_layers = params.get('input_prev_layers')
        alpha_initializer = params.get('alpha_initializer', 'zeros')

        model_rdd = inputRDD(input_prev_layers)
        output_df = PReLULayer(model_rdd, sc, sqlc).add(alpha_initializer)
        outputRDD('<#zzjzRddName#>_activation', output_df)


class ELU(Base):
    def __init__(self, input_prev_layers, alpha):
        super(ELU, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('alpha', alpha)

    def run(self):
        params = self.params

        from tfos.layers import ELULayer

        input_prev_layers = params.get('input_prev_layers')
        alpha = params.get('alpha', '1.0')

        model_rdd = inputRDD(input_prev_layers)
        output_df = ELULayer(model_rdd, sc, sqlc).add(float(alpha))
        outputRDD('<#zzjzRddName#>_activation', output_df)


class ThresholdedReLU(Base):
    def __init__(self, input_prev_layers, theta):
        super(ThresholdedReLU, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('theta', theta)

    def run(self):
        params = self.params

        from tfos.layers import ThresholdedReLULayer

        input_prev_layers = params.get('input_prev_layers')
        theta = params.get('theta', '1.0')

        model_rdd = inputRDD(input_prev_layers)
        output_df = ThresholdedReLULayer(model_rdd, sc, sqlc).add(float(theta))
        outputRDD('<#zzjzRddName#>_activation', output_df)


class Softmax(Base):
    def __init__(self, input_prev_layers, axis):
        super(Softmax, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('axis', axis)

    def run(self):
        params = self.params

        from tfos.layers import SoftmaxLayer

        input_prev_layers = params.get('input_prev_layers')
        axis = params.get('axis', '-1')

        model_rdd = inputRDD(input_prev_layers)
        output_df = SoftmaxLayer(model_rdd, sc, sqlc).add(int(axis))
        outputRDD('<#zzjzRddName#>_activation', output_df)


class ReLU(Base):
    def __init__(self, input_prev_layers, max_value, negative_slope, threshold):
        super(ReLU, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
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
        outputRDD('<#zzjzRddName#>_activation', output_df)


class TestActivation(unittest.TestCase):

    @unittest.skip('')
    def test_activation(self):
        Dense(lrn(), 512, input_dim=784).run()
        Activation(lrn(), activation='relu').run()
        Activation(lrn(), activation='relu').run()
        Activation(lrn(), activation='relu').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_leaky_relu(self):
        Dense(lrn(), 512, input_dim=784).run()
        LeakyReLU(lrn(), alpha='0.3').run()
        LeakyReLU(lrn(), alpha='0.3').run()
        LeakyReLU(lrn(), alpha='0.3').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_prelu(self):
        Dense(lrn(), 512, input_dim=784).run()
        PReLU(lrn(), alpha_initializer='zeros').run()
        PReLU(lrn(), alpha_initializer='zeros').run()
        PReLU(lrn(), alpha_initializer='zeros').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_elu(self):
        Dense(lrn(), 512, input_dim=784).run()
        ELU(lrn(), alpha='1.0').run()
        ELU(lrn(), alpha='1.0').run()
        ELU(lrn(), alpha='1.0').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_thresholded_relu(self):
        Dense(lrn(), 512, input_dim=784).run()
        ThresholdedReLU(lrn(), theta='1.0').run()
        ThresholdedReLU(lrn(), theta='1.0').run()
        ThresholdedReLU(lrn(), theta='1.0').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_softmax(self):
        Dense(lrn(), 512, input_dim=784).run()
        Softmax(lrn(), axis='-1').run()
        Softmax(lrn(), axis='-1').run()
        Softmax(lrn(), axis='-1').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_relu(self):
        Dense(lrn(), 512, input_dim=784).run()
        ReLU(lrn(), max_value='', negative_slope='0', threshold='0').run()
        ReLU(lrn(), max_value='', negative_slope='0', threshold='0').run()
        ReLU(lrn(), max_value='', negative_slope='0', threshold='0').run()
        SummaryLayer(lrn()).run()


if __name__ == "__main__":
    unittest.main()
