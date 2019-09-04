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
from deep_insight.layers.input import InputLayer


class MaxPool1D(Base):
    """1D最大池化层
    对于时序数据的最大池化。

    参数
        pool_size: 窗口大小
            整数，最大池化的窗口大小。
        strides: 收缩因子
            整数，或者是 None。作为缩小比例的因数。 例如，2 会使得输入张量缩小一半。 如果是 None，那么默认值是 pool_size。
        padding: 边界填充
            "valid" 或者 "same" （区分大小写）。
        data_format: 数据格式
            字符串，channels_last (默认)或 channels_first 之一。 表示输入各维度的顺序。 channels_last 对应输入尺寸为
            (batch, steps, features)， channels_first 对应输入尺寸为 (batch, features, steps)。

    输入尺寸
        如果 data_format='channels_last'， 输入为 3D 张量，尺寸为： (batch_size, steps, features)
        如果data_format='channels_first'， 输入为 3D 张量，尺寸为： (batch_size, features, steps)

    输出尺寸
        如果 data_format='channels_last'， 输出为 3D 张量，尺寸为： (batch_size, downsampled_steps, features)
        如果 data_format='channels_first'， 输出为 3D 张量，尺寸为： (batch_size, features, downsampled_steps)
    """

    def __init__(self, pool_size='2', strides=None, padding='valid', data_format='channels_last'):
        super(MaxPool1D, self).__init__()
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import MaxPool1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')
        data_format = param.get('data_format', 'channels_last')

        kwargs = {}

        if pool_size:
            kwargs['pool_size'] = int(pool_size)

        if strides:
            kwargs['strides'] = int(strides)

        if padding:
            kwargs['padding'] = padding

        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = MaxPool1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_MaxPool1D', output_df)


class AvgPool1D(Base):
    """1D均值池化层
    对于时序数据的平均池化。

    参数
        pool_size: 窗口大小
            整数，平均池化的窗口大小。
        strides: 收缩因子
            整数，或者是 None。作为缩小比例的因数。 例如，2 会使得输入张量缩小一半。 如果是 None，那么默认值是 pool_size。
        padding: 边界填充
            "valid" 或者 "same" （区分大小写）。
        data_format: 数据格式
            字符串，channels_last (默认)或 channels_first 之一。 表示输入各维度的顺序。 channels_last 对应输入尺寸
            为 (batch, steps, features)， channels_first 对应输入尺寸为 (batch, features, steps)。

    输入尺寸
        如果 data_format='channels_last'， 输入为 3D 张量，尺寸为： (batch_size, steps, features)
        如果data_format='channels_first'， 输入为 3D 张量，尺寸为： (batch_size, features, steps)

    输出尺寸
        如果 data_format='channels_last'， 输出为 3D 张量，尺寸为： (batch_size, downsampled_steps, features)
        如果 data_format='channels_first'， 输出为 3D 张量，尺寸为： (batch_size, features, downsampled_steps)
    """

    def __init__(self, pool_size='2', strides=None, padding='valid', data_format='channels_last'):
        super(AvgPool1D, self).__init__()
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import AvgPool1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')
        data_format = param.get('data_format', '')

        kwargs = {}

        if pool_size:
            kwargs['pool_size'] = int(pool_size)

        if strides:
            kwargs['strides'] = int(strides)

        if padding:
            kwargs['padding'] = padding

        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_AvgPool1D', output_df)


class MaxPool2D(Base):
    """2D最大池化层
    对于空间数据的最大池化。

    参数
        pool_size: 窗口大小
            整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。
            如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
        strides: 收缩因子
            整数，2 个整数表示的元组，或者是 None。 表示步长值。 如果是 None，那么默认值是 pool_size。
        padding: 边界填充
            "valid" 或者 "same" （区分大小写）。
        data_format: 数据格式
            字符串，channels_last (默认)或 channels_first 之一。 表示输入各维度的顺序。 channels_last 代表尺寸是
            (batch, height, width, channels) 的输入张量， 而 channels_first 代表尺寸是 (batch, channels, height, width)
            的输入张量。 默认值根据 Keras 配置文件 ~/.keras/keras.json 中的 image_data_format 值来设置。 如果还没有设置过，
            那么默认值就是 "channels_last"。

    输入尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, rows, cols, channels) 的 4D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, rows, cols) 的 4D 张量

    输出尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, pooled_rows, pooled_cols, channels) 的 4D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, pooled_rows, pooled_cols) 的 4D 张量
    """

    def __init__(self, pool_size='2,2', strides=None, padding='valid', data_format=None):
        super(MaxPool2D, self).__init__()
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import MaxPool2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')
        data_format = param.get('data_format', '')

        kwargs = {}

        if pool_size:
            pool_size = tuple([int(i) for i in pool_size.split(',') if i])
            assert len(pool_size) == 2, "MaxPool2D pool_size must be 2 dimension!"
            kwargs['pool_size'] = pool_size

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "MaxPool2D strides must be 2 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = MaxPool2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_MaxPool2D', output_df)


class AvgPool2D(Base):
    """2D均值池化层
    对于空间数据的平均池化。

    参数
        pool_size: 窗口大小
            整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。
             如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
        strides: 收缩因子
            整数，2 个整数表示的元组，或者是 None。 表示步长值。 如果是 None，那么默认值是 pool_size。
        padding: 边界填充
            "valid" 或者 "same" （区分大小写）。
        data_format: 数据格式
            字符串，channels_last (默认)或 channels_first 之一。 表示输入各维度的顺序。 channels_last 代表尺寸是
             (batch, height, width, channels) 的输入张量， 而 channels_first 代表尺寸是 (batch, channels, height, width)
             的输入张量。 默认值根据 Keras 配置文件 ~/.keras/keras.json 中的 image_data_format 值来设置。 如果还没有设置过，
             那么默认值就是 "channels_last"。

    输入尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, rows, cols, channels) 的 4D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, rows, cols) 的 4D 张量

    输出尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, pooled_rows, pooled_cols, channels) 的 4D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, pooled_rows, pooled_cols) 的 4D 张量
    """

    def __init__(self, pool_size='2,2', strides=None, padding='valid', data_format=None):
        super(AvgPool2D, self).__init__()
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import AvgPool2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')
        data_format = param.get('data_format', '')

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

        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_AvgPool2D', output_df)


class MaxPool3D(Base):
    """3D最大池化层
    对于 3D（空间，或时空间）数据的最大池化。

    参数
        pool_size: 窗口大小
            3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。 (2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
        strides: 收缩因子
            3 个整数表示的元组，或者是 None。步长值。
        padding: 边界填充
            "valid" 或者 "same"（区分大小写）。
        data_format: 数据格式
            字符串，channels_last (默认)或 channels_first 之一。 表示输入各维度的顺序。 channels_last 代表尺寸是
             (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) 的输入张量， 而 channels_first 代表
             尺寸是 (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3) 的输入张量。 默认值根据 Keras
             配置文件 ~/.keras/keras.json 中的 image_data_format 值来设置。 如果还没有设置过，那么默认值就是 "channels_last"。

    输入尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels) 的 5D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3) 的 5D 张量

    输出尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels) 的 5D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3) 的 5D 张量
    """

    def __init__(self, pool_size='2,2,2', strides=None, padding='valid', data_format=None):
        super(MaxPool3D, self).__init__()
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import AvgPool3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')
        data_format = param.get('data_format', '')

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

        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_MaxPool3D', output_df)


class AvgPool3D(Base):
    """3D均值池化层
    对于 3D （空间，或者时空间）数据的平均池化。

    参数
        pool_size: 窗口大小
            3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。 (2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
        strides: 收缩因子
            3 个整数表示的元组，或者是 None。步长值。
        padding: 边界填充
            "valid" 或者 "same"（区分大小写）。
        data_format: 数据格式
            字符串，channels_last (默认)或 channels_first 之一。 表示输入各维度的顺序。 channels_last 代表尺
            寸是 (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels) 的输入张量， 而 channels_first
             代表尺寸是 (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3) 的输入张量。 默认值根据 
             Keras 配置文件 ~/.keras/keras.json 中的 image_data_format 值来设置。 如果还没有设置过，那么默认
             值就是 "channels_last"。
    
    输入尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels) 的 5D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3) 的 5D 张量
    
    输出尺寸
        如果 data_format='channels_last': 尺寸是 (batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels) 的 5D 张量
        如果 data_format='channels_first': 尺寸是 (batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3) 的 5D 张量
    """

    def __init__(self, pool_size='2,2,2', strides=None, padding='valid', data_format=None):
        super(AvgPool3D, self).__init__()
        self.p('pool_size', pool_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import AvgPool3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        pool_size = param.get('pool_size', '2,2,2')  # integer
        strides = param.get('strides', '')
        padding = param.get('padding', 'valid')
        data_format = param.get('data_format', '')

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

        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = AvgPool3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_AvgPool3D', output_df)


class TestPool(unittest.TestCase):
    def tearDown(self) -> None:
        reset()

    # @unittest.skip('')
    def test_max_pool1d(self):
        Convolution1D(filters=32, kernel_size='3', strides='1', input_shape='100,3').run()
        MaxPool1D('2', '', 'valid').run()
        MaxPool1D('2', '', 'valid').run()
        MaxPool1D('2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_avg_pool1d(self):
        Convolution1D(filters=32, kernel_size='3', strides='1', input_shape='100,3').run()
        AvgPool1D('2', '', 'valid').run()
        AvgPool1D('2', '', 'valid').run()
        AvgPool1D('2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_max_pool2d(self):
        Convolution2D(filters=32, kernel_size='3,3', strides='1,1', input_shape='100,100,3').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_avg_pool2d(self):
        Convolution2D(filters=32, kernel_size='3,3', strides='1,1', input_shape='100,100,3').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_max_pool3d(self):
        Convolution3D(filters=32, kernel_size='3,3,3', strides='1,1,1', input_shape='100,100,100,3').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        SummaryLayer().run()


class TestPoolNetWork(unittest.TestCase):
    def tearDown(self) -> None:
        reset()

    # @unittest.skip('')
    def test_max_pool1d(self):
        InputLayer('100,3').run()
        Convolution1D(filters=32, kernel_size='3', strides='1').run()
        MaxPool1D('2', '', 'valid').run()
        MaxPool1D('2', '', 'valid').run()
        MaxPool1D('2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_avg_pool1d(self):
        InputLayer('100,3').run()
        Convolution1D(filters=32, kernel_size='3', strides='1').run()
        AvgPool1D('2', '', 'valid').run()
        AvgPool1D('2', '', 'valid').run()
        AvgPool1D('2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_max_pool2d(self):
        InputLayer('100,100,3').run()
        Convolution2D(filters=32, kernel_size='3,3', strides='1,1').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_avg_pool2d(self):
        InputLayer('100,100,3').run()
        Convolution2D(filters=32, kernel_size='3,3', strides='1,1').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        MaxPool2D('2,2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_max_pool3d(self):
        InputLayer('100,100,100,3').run()
        Convolution3D(filters=32, kernel_size='3,3,3', strides='1,1,1').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_avg_pool3d(self):
        InputLayer('100,100,100,3').run()
        Convolution3D(filters=32, kernel_size='3,3,3', strides='1,1,1').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        MaxPool3D('2,2,2', '', 'valid').run()
        SummaryLayer().run()


if __name__ == '__main__':
    unittest.main()
