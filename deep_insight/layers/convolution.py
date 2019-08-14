#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 12:57
:File       : convolution.py
"""
import unittest
from deep_insight.base import *


class Convolution(Base):
    def __init__(self, input_prev_layers, filters, kernel_size, strides, padding=None, activation=None,
                 input_shape=None):
        super(Convolution, self).__init__()
        self.p('input_prev_layers', input_prev_layers)
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('activation', activation)
        self.p('input_shape', input_shape)

    def run(self):
        raise NotImplementedError


class Convolution1D(Convolution):
    """一维卷积算子

    参数：
        input_prev_layers: 输入模型
            当前算子层之前构建的模型层参数
        filters: 过滤器数量
            正整数，每一个过滤器输出一个结果，当前层输出为过滤器数量大小的维度
        kernel_size: 卷积核
            一维卷积核为一维的正整数
        strides: 移动窗口
            一维卷积核为一维的正整数
        padding: 填充方式
            进行卷积运算时边缘的填充方式，valid表示不填充，same表示填充
        activation: 激活函数
            当前算子的激活函数，默认值为空
        input_shape： 输入形状
            输入数据的维度形状，当前为第一层模型算子时，该参数不能为空，其余情况可为空
    """
    def run(self):
        param = self.params

        from tfos.choices import CHOICES
        from tfos.layers import Conv1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')  # integer
        kernel_size = param.get('kernel_size')  # two integer separate with a comma
        strides = param.get('strides', '1')  # two integer separate with a comma
        padding = param.get('padding', CHOICES['padding'][0])
        activation = param.get('activation', CHOICES['activation'][0])
        input_shape = param.get('input_shape', '')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 1, "parameter kernel_size must be 1 dimension!"
        kwargs = {
            "filters": int(filters),
            "kernel_size": kernel_size
        }

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 1, "parameter strides must be 1 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        if activation:
            kwargs['activation'] = activation

        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution2D(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = int(param.get('filters'))  # integer
        kernel_size = param.get('kernel_size')  # two integer separate with a comma
        strides = param.get('strides')  # two integer separate with a comma
        padding = param.get('padding')
        activation = param.get('activation')
        input_shape = param.get('input_shape')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 2, "parameter kernel_size must be 2 dimension!"
        kwargs = {
            "filters": int(filters),
            "kernel_size": kernel_size
        }

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "parameter strides must be 2 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        if activation:
            kwargs['activation'] = activation

        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution3D(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')  # integer
        kernel_size = param.get('kernel_size')  # two integer separate with a comma
        strides = param.get('strides')  # two integer separate with a comma
        padding = param.get('padding')
        activation = param.get('activation')
        input_shape = param.get('input_shape')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 3, "parameter kernel_size must be 3 dimension!"
        kwargs = {
            "filters": int(filters),
            "kernel_size": kernel_size
        }

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 3, "parameter strides must be 3 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        if activation:
            kwargs['activation'] = activation

        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution2DTranspose(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv2DTransposeLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')  # integer
        kernel_size = param.get('kernel_size')  # two integer separate with a comma
        strides = param.get('strides')  # two integer separate with a comma
        padding = param.get('padding')
        activation = param.get('activation')
        input_shape = param.get('input_shape')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 2, "parameter kernel_size must be 2 dimension!"
        kwargs = {
            "filters": int(filters),
            "kernel_size": kernel_size
        }

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "parameter strides must be 2 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        if activation:
            kwargs['activation'] = activation

        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv2DTransposeLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution3DTranspose(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv3DTransposeLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')  # integer
        kernel_size = param.get('kernel_size')  # two integer separate with a comma
        strides = param.get('strides')  # two integer separate with a comma
        padding = param.get('padding')
        activation = param.get('activation')
        input_shape = param.get('input_shape')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 3, "parameter kernel_size must be 3 dimension!"
        kwargs = {
            "filters": int(filters),
            "kernel_size": kernel_size
        }

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 3, "parameter strides must be 3 dimension!"
            kwargs['strides'] = strides

        if padding:
            kwargs['padding'] = padding

        if activation:
            kwargs['activation'] = activation

        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv3DTransposeLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class TestConvolution(unittest.TestCase):
    @unittest.skip('')
    def test_conv1d(self):
        Convolution1D(lrn(), filters=32, kernel_size='3', strides='1', input_shape='100,3').run()
        Convolution1D(lrn(), filters=64, kernel_size='3', strides='1').run()
        Convolution1D(lrn(), filters=256, kernel_size='3', strides='1').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_conv2d(self):
        Convolution2D(lrn(), filters=32, kernel_size='3,3', strides='1,1', input_shape='100,100,3').run()
        Convolution2D(lrn(), filters=64, kernel_size='3,3', strides='1,1').run()
        Convolution2D(lrn(), filters=256, kernel_size='3,3', strides='1,1').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_conv3d(self):
        Convolution3D(lrn(), filters=32, kernel_size='3,3,3', strides='1,1,1', input_shape='100,100,100,3').run()
        Convolution3D(lrn(), filters=64, kernel_size='3,3,3', strides='1,1,1').run()
        Convolution3D(lrn(), filters=256, kernel_size='3,3,3', strides='1,1,1').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_conv2d_transpose(self):
        Convolution2DTranspose(lrn(), filters=32, kernel_size='3,3', strides='1,1', input_shape='100,100,3').run()
        Convolution2DTranspose(lrn(), filters=64, kernel_size='3,3', strides='1,1').run()
        Convolution2DTranspose(lrn(), filters=256, kernel_size='3,3', strides='1,1').run()
        SummaryLayer(lrn()).run()

    @unittest.skip('')
    def test_conv3d_transpose(self):
        Convolution3DTranspose(lrn(), filters=32, kernel_size='3,3,3', strides='1,1,1',
                               input_shape='100,100,100,3').run()
        Convolution3DTranspose(lrn(), filters=64, kernel_size='3,3,3', strides='1,1,1').run()
        Convolution3DTranspose(lrn(), filters=256, kernel_size='3,3,3', strides='1,1,1').run()
        SummaryLayer(lrn()).run()


if __name__ == "__main__":
    unittest.main()
