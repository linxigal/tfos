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
    def __init__(self, input_model_config_name, filters, kernel_size, strides, padding=None, activation=None,
                 input_shape=None):
        super(Convolution, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('activation', activation)
        self.p('input_shape', input_shape)

    def run(self):
        raise NotImplementedError


class Convolution1D(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        filters = param.get('filters')  # integer
        kernel_size = param.get('kernel_size')  # two integer separate with a comma
        strides = param.get('strides', '1')  # two integer separate with a comma
        padding = param.get('padding', 'valid')
        activation = param.get('activation', '')
        input_shape = param.get('input_shape')  # many integer separate with comma

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

        model_rdd = inputRDD(input_model_config_name)
        output_df = Conv1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution2D(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
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

        model_rdd = inputRDD(input_model_config_name)
        output_df = Conv2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution3D(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
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

        model_rdd = inputRDD(input_model_config_name)
        output_df = Conv3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution2DTranspose(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv2DTransposeLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
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

        model_rdd = inputRDD(input_model_config_name)
        output_df = Conv2DTransposeLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution3DTranspose(Convolution):
    def run(self):
        param = self.params

        from tfos.layers import Conv3DTransposeLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
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

        model_rdd = inputRDD(input_model_config_name)
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
