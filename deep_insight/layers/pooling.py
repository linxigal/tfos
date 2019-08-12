#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/12 15:56
:File   : pooling.py
"""

import unittest

from deep_insight.base import *
from deep_insight.layers.convolution import Convolution1D, Convolution2D, Convolution3D


class Pool(Base):
    def __init__(self, input_prev_layers, pool_size, strides, padding):
        super(Pool, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)

    def run(self):
        raise NotImplementedError


class MaxPool1D(Pool):
    def run(self):
        param = self.params

        from tfos.layers import MaxPool1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')

        kwargs = {}

        if pool_size:
            kwargs['pool_size'] = int(pool_size)

        if strides:
            kwargs['strides'] = int(strides)

        if padding:
            kwargs['padding'] = padding

        model_rdd = inputRDD(input_prev_layers)
        output_df = MaxPool1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Pool', output_df)


class AvgPool1D(Pool):
    def run(self):
        param = self.params

        from tfos.layers import AvgPool1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')

        kwargs = {}

        if pool_size:
            kwargs['pool_size'] = int(pool_size)

        if strides:
            kwargs['strides'] = int(strides)

        if padding:
            kwargs['padding'] = padding

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Pool', output_df)


class MaxPool2D(Pool):
    def run(self):
        param = self.params

        from tfos.layers import MaxPool2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')

        kwargs = {}

        if pool_size:
            pool_size = tuple([int(i) for i in pool_size.split(',') if i])
            assert len(pool_size) == 2, "parameter pool_size must be 2 dimension!"
            kwargs['pool_size'] = pool_size

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "parameter strides must be 2 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        model_rdd = inputRDD(input_prev_layers)
        output_df = MaxPool2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Pool', output_df)


class AvgPool2D(Pool):
    def run(self):
        param = self.params

        from tfos.layers import AvgPool2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')

        kwargs = {}

        if pool_size:
            pool_size = tuple([int(i) for i in pool_size.split(',') if i])
            assert len(pool_size) == 2, "parameter pool_size must be 2 dimension!"
            kwargs['pool_size'] = pool_size

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "parameter strides must be 2 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Pool', output_df)


class MaxPool3D(Pool):
    def run(self):
        param = self.params

        from tfos.layers import AvgPool3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')

        kwargs = {}

        if pool_size:
            pool_size = tuple([int(i) for i in pool_size.split(',') if i])
            assert len(pool_size) == 3, "parameter pool_size must be 3 dimension!"
            kwargs['pool_size'] = pool_size

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 3, "parameter strides must be 3 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Pool', output_df)


class AvgPool3D(Pool):
    def run(self):
        param = self.params

        from tfos.layers import AvgPool3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')

        kwargs = {}

        if pool_size:
            pool_size = tuple([int(i) for i in pool_size.split(',') if i])
            assert len(pool_size) == 3, "parameter pool_size must be 3 dimension!"
            kwargs['pool_size'] = pool_size

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 3, "parameter strides must be 3 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Pool', output_df)


class TestPool(unittest.TestCase):
    @unittest.skip('')
    def test_max_pool1d(self):
        Convolution1D(lrn(), filters=32, kernel_size='3', strides='1', input_shape='100,3').run()
        MaxPool1D(lrn(), '2', '', 'valid').run()
        MaxPool1D(lrn(), '2', '', 'valid').run()
        MaxPool1D(lrn(), '2', '', 'valid').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_avg_pool1d(self):
        Convolution1D(lrn(), filters=32, kernel_size='3', strides='1', input_shape='100,3').run()
        AvgPool1D(lrn(), '2', '', 'valid').run()
        AvgPool1D(lrn(), '2', '', 'valid').run()
        AvgPool1D(lrn(), '2', '', 'valid').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_max_pool2d(self):
        Convolution2D(lrn(), filters=32, kernel_size='3,3', strides='1,1', input_shape='100,100,3').run()
        MaxPool2D(lrn(), '2,2', '', 'valid').run()
        MaxPool2D(lrn(), '2,2', '', 'valid').run()
        MaxPool2D(lrn(), '2,2', '', 'valid').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_avg_pool2d(self):
        Convolution2D(lrn(), filters=32, kernel_size='3,3', strides='1,1', input_shape='100,100,3').run()
        MaxPool2D(lrn(), '2,2', '', 'valid').run()
        MaxPool2D(lrn(), '2,2', '', 'valid').run()
        MaxPool2D(lrn(), '2,2', '', 'valid').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_max_pool3d(self):
        Convolution3D(lrn(), filters=32, kernel_size='3,3,3', strides='1,1,1', input_shape='100,100,100,3').run()
        MaxPool3D(lrn(), '2,2,2', '', 'valid').run()
        MaxPool3D(lrn(), '2,2,2', '', 'valid').run()
        MaxPool3D(lrn(), '2,2,2', '', 'valid').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_avg_pool3d(self):
        Convolution3D(lrn(), filters=32, kernel_size='3,3,3', strides='1,1,1', input_shape='100,100,100,3').run()
        MaxPool3D(lrn(), '2,2,2', '', 'valid').run()
        MaxPool3D(lrn(), '2,2,2', '', 'valid').run()
        MaxPool3D(lrn(), '2,2,2', '', 'valid').run()
        SummaryLayer(lrn()).run()


if __name__ == '__main__':
    unittest.main()
