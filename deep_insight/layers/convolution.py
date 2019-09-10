#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 12:57
:File       : convolution.py
"""
import unittest

from deep_insight.base import *
from deep_insight.layers.input import InputLayer


class Convolution1D(Base):
    """1D卷积层
    1D 卷积层 (例如时序卷积)。

    该层创建了一个卷积核，该卷积核以 单个空间（或时间）维上的层输入进行卷积， 以生成输出张量。 如果 use_bias 为 True，
    则会创建一个偏置向量并将其添加到输出中。 最后，如果 activation 不是 None，它也会应用于输出。

    当使用该层作为模型第一层时，需要提供 input_shape 参数（整数元组或 None），例如， (10, 128) 表示 10 个 128 维的向
    量组成的向量序列， (None, 128) 表示 128 维的向量组成的变长序列。

    参数:
        filters: 输出空间
            整数，输出空间的维度 （即卷积中滤波器的输出数量）。
        kernel_size: 卷积核
            一个整数，或者单个整数表示的元组或列表， 指明 1D 卷积窗口的长度。
        strides: 移动步长
            一个整数，或者单个整数表示的元组或列表， 指明卷积的步长。 指定任何 stride 值 != 1 与指定 dilation_rate
            值 != 1 两者不兼容。多个值用英文逗号分隔
        padding: 边界填充
            "valid", "causal" 或 "same" 之一 (大小写敏感) "valid" 表示「不填充」。 "same" 表示填充输入以使输出具有
            与原始输入相同的长度。 "causal" 表示因果（膨胀）卷积， 例如，output[t] 不依赖于 input[t+1:]， 在模型不应
            违反时间顺序的时间数据建模时非常有用。 详见 WaveNet: A Generative Model for Raw Audio, section 2.1。
        data_format: 数据格式
            字符串, "channels_last" (默认) 或 "channels_first" 之一。输入的各个维度顺序。 "channels_last" 对应输入
            尺寸为 (batch, steps, channels) (Keras 中时序数据的默认格式) 而 "channels_first" 对应输入尺寸
            为 (batch, channels, steps)。
        dilation_rate: 膨胀率
            一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。 当前，指定任何 dilation_rate 值 != 1
            与指定 stride 值 != 1 两者不兼容。多个值用英文逗号分隔
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 如未指定，则不使用激活函数 (即线性激活： a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 卷积核初始化
            kernel 权值矩阵的初始化器 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见 initializers)。
        kernel_regularizer: 卷积核正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 激活值正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 卷积核约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。

    输入尺寸
        3D 张量 ，尺寸为 (batch_size, steps, input_dim)。

    输出尺寸
        3D 张量，尺寸为 (batch_size, new_steps, filters)。 由于填充或窗口按步长滑动，steps 值可能已更改。
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides='1',
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate='1',
                 activation=None,
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None):
        super(Convolution1D, self).__init__()
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)
        self.p('dilation_rate', dilation_rate)
        self.p('activation', activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import Conv1DLayer
        from tfos.choices import PADDING, ACTIVATIONS, BOOLEAN

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')
        kernel_size = param.get('kernel_size')
        strides = param.get('strides', '1')
        padding = param.get('padding', PADDING[0])
        data_format = param.get('data_format', 'channels_last')
        dilation_rate = param.get('dilation_rate', '1,1')
        activation = param.get('activation', ACTIVATIONS[0])
        use_bias = param.get('use_bias', BOOLEAN[0])
        kernel_initializer = param.get('kernel_initializer', 'glorot_uniform')
        bias_initializer = param.get('bias_initializer', 'zeros')
        kernel_regularizer = param.get('kernel_regularizer', '')
        bias_regularizer = param.get('bias_regularizer', '')
        activity_regularizer = param.get('activity_regularizer', '')
        kernel_constraint = param.get('kernel_constraint', '')
        bias_constraint = param.get('bias_constraint', '')
        input_shape = param.get('input_shape', '')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 1, "Convolution1D kernel_size must be 1 dimension!"
        kwargs = dict(filters=int(filters), kernel_size=kernel_size)

        if strides:
            kwargs['strides'] = int(strides)
        if padding:
            kwargs['padding'] = padding
        if data_format:
            kwargs['data_format'] = data_format
        if dilation_rate:
            kwargs['dilation_rate'] = int(dilation_rate)
        if activation:
            kwargs['activation'] = activation
        if use_bias:
            kwargs['use_bias'] = True if use_bias.lower() == 'true' else False
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Convolution1D', output_df)


class Convolution2D(Base):
    """2D卷积
    2D 卷积层 (例如对图像的空间卷积)。

    该层创建了一个卷积核， 该卷积核对层输入进行卷积， 以生成输出张量。 如果 use_bias 为 True， 则会创建一个偏置向量
    并将其添加到输出中。 最后，如果 activation 不是 None，它也会应用于输出。

    当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含样本表示的轴），例如，
    input_shape=(128, 128, 3) 表示 128x128 RGB 图像， 在 data_format="channels_last" 时。

    参数:
        filters: 输出空间
            整数，输出空间的维度 （即卷积中滤波器的输出数量）。
        kernel_size: 卷积核
            一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，
            为所有空间维度指定相同的值。
        strides: 移动步长
            一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，
            为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
            多个值用英文逗号分隔
        padding: 边界填充
            "valid" 或 "same" (大小写敏感)。
        data_format: 数据格式
            字符串， channels_last (默认) 或 channels_first 之一，表示输入中维度的顺序。 channels_last 对应输入尺寸
            为 (batch, height, width, channels)， channels_first 对应输入尺寸为 (batch, channels, height, width)。
            它默认为从 Keras 配置文件 ~/.keras/keras.json 中 找到的 image_data_format 值。 如果你从未设置它，
            将使用 channels_last。
        dilation_rate: 膨胀率
            一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。
            当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 两者不兼容。多个值用英文逗号分隔
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 卷积核初始化
            kernel 权值矩阵的初始化器 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见 initializers)。
        kernel_regularizer: 卷积核正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 激活值正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 卷积核约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。

    输入尺寸
        如果 data_format='channels_first'， 输入 4D 张量，尺寸为 (samples, channels, rows, cols)。
        如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (samples, rows, cols, channels)。

    输出尺寸
        如果 data_format='channels_first'， 输出 4D 张量，尺寸为 (samples, filters, new_rows, new_cols)。
        如果 data_format='channels_last'， 输出 4D 张量，尺寸为 (samples, new_rows, new_cols, filters)。

    由于填充的原因， rows 和 cols 值可能已更改。
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides='1,1',
                 padding='valid',
                 data_format=None,
                 dilation_rate='1,1',
                 activation=None,
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None):
        super(Convolution2D, self).__init__()
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)
        self.p('dilation_rate', dilation_rate)
        self.p('activation', activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import Conv2DLayer
        from tfos.choices import D_ACTIVATIONS, BOOLEAN, PADDING

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')
        kernel_size = param.get('kernel_size')
        strides = param.get('strides', '1,1')
        padding = param.get('padding', PADDING[0])
        data_format = param.get('data_format', '')
        dilation_rate = param.get('dilation_rate', '1,1')
        activation = param.get('activation', D_ACTIVATIONS[0])
        use_bias = param.get('use_bias', BOOLEAN[0])
        kernel_initializer = param.get('kernel_initializer', 'glorot_uniform')
        bias_initializer = param.get('bias_initializer', 'zeros')
        kernel_regularizer = param.get('kernel_regularizer', '')
        bias_regularizer = param.get('bias_regularizer', '')
        activity_regularizer = param.get('activity_regularizer', '')
        kernel_constraint = param.get('kernel_constraint', '')
        bias_constraint = param.get('bias_constraint', '')
        input_shape = param.get('input_shape', '')

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 2, "Convolution2D kernel_size must be 2 dimension!"
        kwargs = dict(filters=int(filters), kernel_size=kernel_size)

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "Convolution2D strides must be 2 dimension!"
            kwargs['strides'] = strides
        if padding:
            kwargs['padding'] = padding
        if data_format:
            kwargs['data_format'] = data_format
        if dilation_rate:
            dilation_rate = tuple([int(i) for i in dilation_rate.split(',') if i])
            assert len(dilation_rate) == 2, "parameter strides must be 2 dimension!"
            kwargs['dilation_rate'] = dilation_rate
        if activation:
            kwargs['activation'] = activation
        if use_bias:
            kwargs['use_bias'] = True if use_bias.lower() == 'true' else False
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution3D(Base):
    """3D卷积层
    3D 卷积层 (例如立体空间卷积)。

    该层创建了一个卷积核， 该卷积核对层输入进行卷积， 以生成输出张量。 如果 use_bias 为 True， 则会创建一个偏置向量并将其
    添加到输出中。 最后，如果 activation 不是 None，它也会应用于输出。

    当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含样本表示的轴），例如，
    input_shape=(128, 128, 128, 1) 表示 128x128x128 的单通道立体， 在 data_format="channels_last" 时。

    参数:
        filters: 输出空间
            整数，输出空间的维度 （即卷积中滤波器的输出数量）。
        kernel_size: 卷积核
            一个整数，或者 3 个整数表示的元组或列表， 指明 3D 卷积窗口的深度、高度和宽度。 可以是一个整数，
            为所有空间维度指定相同的值。
        strides: 移动步长
            一个整数，或者 3 个整数表示的元组或列表， 指明卷积沿每一个空间维度的步长。 可以是一个整数，为所有空间维度指定
            相同的步长值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。多个值用英文逗号分隔
        padding: 边界填充
            "valid" 或 "same" (大小写敏感)。
        data_format: 数据格式
            字符串， channels_last (默认) 或 channels_first 之一， 表示输入中维度的顺序。channels_last
            对应输入尺寸为 (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)， channels_first 对应输入
            尺寸为 (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)。 它默认为从 Keras 配置文件
            ~/.keras/keras.json 中 找到的 image_data_format 值。 如果你从未设置它，将使用 "channels_last"。
        dilation_rate: 膨胀率
            一个整数或 3 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。
            当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 两者不兼容。多个值用英文逗号分隔
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 卷积核初始化
            kernel 权值矩阵的初始化器 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见 initializers)。
        kernel_regularizer: 卷积核正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 激活值正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 卷积核约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。

    输入尺寸
        如果 data_format='channels_first'， 输入 5D 张量，尺寸为 (samples, channels, conv_dim1, conv_dim2, conv_dim3)。
        如果 data_format='channels_last'， 输入 5D 张量，尺寸为 (samples, conv_dim1, conv_dim2, conv_dim3, channels)。

    输出尺寸
        如果 data_format='channels_first'， 输出 5D 张量，尺寸为 (samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)。
        如果 data_format='channels_last'， 输出 5D 张量，尺寸为 (samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)。
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides='1,1,1',
                 padding='valid',
                 data_format=None,
                 dilation_rate='1,1,1',
                 activation=None,
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None):
        super(Convolution3D, self).__init__()
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('data_format', data_format)
        self.p('dilation_rate', dilation_rate)
        self.p('activation', activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import Conv3DLayer
        from tfos.choices import ACTIVATIONS, BOOLEAN, PADDING

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')
        kernel_size = param.get('kernel_size')
        strides = param.get('strides', '1,1,1')
        padding = param.get('padding', PADDING[0])
        data_format = param.get('data_format', '')
        dilation_rate = param.get('dilation_rate', '1,1')
        activation = param.get('activation', ACTIVATIONS[0])
        use_bias = param.get('use_bias', BOOLEAN[0])
        kernel_initializer = param.get('kernel_initializer', 'glorot_uniform')
        bias_initializer = param.get('bias_initializer', 'zeros')
        kernel_regularizer = param.get('kernel_regularizer', '')
        bias_regularizer = param.get('bias_regularizer', '')
        activity_regularizer = param.get('activity_regularizer', '')
        kernel_constraint = param.get('kernel_constraint', '')
        bias_constraint = param.get('bias_constraint', '')
        input_shape = param.get('input_shape')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 3, "Convolution3D kernel_size must be 3 dimension!"
        kwargs = dict(filters=int(filters), kernel_size=kernel_size)

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 3, "Convolution3D strides must be 3 dimension!"
            kwargs['strides'] = strides
        if padding:
            kwargs['padding'] = padding
        if data_format:
            kwargs['data_format'] = data_format
        if dilation_rate:
            dilation_rate = tuple([int(i) for i in dilation_rate.split(',') if i])
            assert len(dilation_rate) == 3, "parameter strides must be 3 dimension!"
            kwargs['dilation_rate'] = dilation_rate
        if activation:
            kwargs['activation'] = activation
        if use_bias:
            kwargs['use_bias'] = True if use_bias.lower() == 'true' else False
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution2DTranspose(Base):
    """2D转置卷积
    转置卷积层 (有时被成为反卷积)。

    对转置卷积的需求一般来自希望使用 与正常卷积相反方向的变换， 即，将具有卷积输出尺寸的东西 转换为具有卷积输入尺寸的东西，
    同时保持与所述卷积相容的连通性模式。

    当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含样本表示的轴），例如， input_shape=(128, 128, 3)
    表示 128x128 RGB 图像， 在 data_format="channels_last" 时。

    参数:
        filters: 输出空间
            整数，输出空间的维度 （即卷积中滤波器的输出数量）。
        kernel_size: 卷积核
         一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的高度和宽度。 可以是一个整数，
        为所有空间维度指定相同的值。
        strides: 移动步长
            一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿高度和宽度方向的步长。 可以是一个整数，
            为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
            多个值用英文逗号分隔
        padding: 边界填充
            "valid" 或 "same" (大小写敏感)。
        output_padding: 输出填充
            一个整数，或者 2 个整数表示的元组或列表， 指定沿输出张量的高度和宽度的填充量。 可以是单个整数，
            以指定所有空间维度的相同值。 沿给定维度的输出填充量必须低于沿同一维度的步长。 如果设置为 None (默认),
            输出尺寸将自动推理出来。
        data_format: 数据格式
            字符串， channels_last (默认) 或 channels_first 之一，表示输入中维度的顺序。 channels_last 对应输入
            尺寸为 (batch, height, width, channels)， channels_first 对应输入尺寸为 (batch, channels, height, width)。
            它默认为从 Keras 配置文件 ~/.keras/keras.json 中 找到的 image_data_format 值。 如果你从未设置它，
            将使用 "channels_last"。
        dilation_rate: 膨胀率
            一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。
            当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 两者不兼容。多个值用英文逗号分隔
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 卷积核初始化
            kernel 权值矩阵的初始化器 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见 initializers)。
        kernel_regularizer: 卷积核正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 激活值正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 卷积核约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。

    输入尺寸
        如果 data_format='channels_first'， 输入 4D 张量，尺寸为 (batch, channels, rows, cols)。
        如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (batch, rows, cols, channels)。

    输出尺寸
        如果 data_format='channels_first'， 输出 4D 张量，尺寸为 (batch, filters, new_rows, new_cols)。
        如果 data_format='channels_last'， 输出 4D 张量，尺寸为 (batch, new_rows, new_cols, filters)。
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides='1,1',
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate='1,1',
                 activation=None,
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None):
        super(Convolution2DTranspose, self).__init__()
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('output_padding', output_padding)
        self.p('data_format', data_format)
        self.p('dilation_rate', dilation_rate)
        self.p('activation', activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import Conv2DTransposeLayer
        from tfos.choices import ACTIVATIONS, BOOLEAN, PADDING

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')
        kernel_size = param.get('kernel_size')
        strides = param.get('strides', '1,1')
        padding = param.get('padding', PADDING[0])
        output_padding = param.get('output_padding', '')
        data_format = param.get('data_format', 'channels_last')
        dilation_rate = param.get('dilation_rate', '1,1')
        activation = param.get('activation', ACTIVATIONS[0])
        use_bias = param.get('use_bias', BOOLEAN[0])
        kernel_initializer = param.get('kernel_initializer', 'glorot_uniform')
        bias_initializer = param.get('bias_initializer', 'zeros')
        kernel_regularizer = param.get('kernel_regularizer', '')
        bias_regularizer = param.get('bias_regularizer', '')
        activity_regularizer = param.get('activity_regularizer', '')
        kernel_constraint = param.get('kernel_constraint', '')
        bias_constraint = param.get('bias_constraint', '')
        input_shape = param.get('input_shape')

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 2, "Convolution2DTranspose kernel_size must be 2 dimension!"
        kwargs = dict(filters=int(filters), kernel_size=kernel_size)

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 2, "parameter strides must be 2 dimension!"
            kwargs['strides'] = strides
        if padding:
            kwargs['padding'] = padding
        if output_padding:
            output_padding = tuple([int(i) for i in output_padding.split(',') if i])
            assert len(output_padding) == 2, "Convolution2DTranspose output_padding must be 2 dimension!"
            kwargs['output_padding'] = output_padding
        if data_format:
            kwargs['data_format'] = data_format
        if dilation_rate:
            dilation_rate = tuple([int(i) for i in dilation_rate.split(',') if i])
            assert len(dilation_rate) == 2, "parameter strides must be 2 dimension!"
            kwargs['dilation_rate'] = dilation_rate
        if activation:
            kwargs['activation'] = activation
        if use_bias:
            kwargs['use_bias'] = True if use_bias.lower() == 'true' else False
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv2DTransposeLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class Convolution3DTranspose(Base):
    """3D转置卷积层
    转置卷积层 (有时被成为反卷积)。

    对转置卷积的需求一般来自希望使用 与正常卷积相反方向的变换， 即，将具有卷积输出尺寸的东西 转换为具有卷积输入尺寸的东西，
     同时保持与所述卷积相容的连通性模式。

    当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含样本表示的轴），例如，
    input_shape=(128, 128, 128, 3) 表示尺寸 128x128x128 的 3 通道立体， 在 data_format="channels_last" 时。

    参数:
        filters: 输出空间
            整数，输出空间的维度 （即卷积中滤波器的输出数量）。
        kernel_size: 卷积核
            一个整数，或者 3 个整数表示的元组或列表， 指明 3D 卷积窗口的深度、高度和宽度。 可以是一个整数，
            为所有空间维度指定相同的值。
        strides: 移动步长
            一个整数，或者 3 个整数表示的元组或列表， 指明沿深度、高度和宽度方向的步长。 可以是一个整数，
            为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
            多个值用英文逗号分隔
        padding: 边界填充
            "valid" 或 "same" (大小写敏感)。
        output_padding: 输出填充
            一个整数，或者 3 个整数表示的元组或列表， 指定沿输出张量的高度和宽度的填充量。 可以是单个整数，
            以指定所有空间维度的相同值。 沿给定维度的输出填充量必须低于沿同一维度的步长。 如果设置为 None (默认),
            输出尺寸将自动推理出来。
        data_format: 数据格式
            字符串， channels_last (默认) 或 channels_first 之一，表示输入中维度的顺序。 channels_last 对应输入
            尺寸为 (batch, depth, height, width, channels)， channels_first 对应输入尺寸为 (batch, channels,
             depth, height, width)。 它默认为从 Keras 配置文件 ~/.keras/keras.json 中 找到的 image_data_format 值。
              如果你从未设置它，将使用「channels_last」。
        dilation_rate: 膨胀率
            一个整数或 3 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。
             当前，指定任何 dilation_rate 值 != 1 与 指定 stride 值 != 1 两者不兼容。多个值用英文逗号分隔
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 卷积核初始化
            kernel 权值矩阵的初始化器 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见 initializers)。
        kernel_regularizer: 卷积核正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 激活值正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 卷积核约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。

    输入尺寸
        如果 data_format='channels_first'， 输入 5D 张量，尺寸为 (batch, channels, depth, rows, cols)，
        如果 data_format='channels_last'， 输入 5D 张量，尺寸为 (batch, depth, rows, cols, channels)。

    Output shape
        如果 data_format='channels_first'， 输出 5D 张量，尺寸为 (batch, filters, new_depth, new_rows, new_cols)，
        如果 data_format='channels_last'， 输出 5D 张量，尺寸为 (batch, new_depth, new_rows, new_cols, filters)。
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides='1,1,1',
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 activation=None,
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None):
        super(Convolution3DTranspose, self).__init__()
        self.p('filters', filters)
        self.p('kernel_size', kernel_size)
        self.p('strides', strides)
        self.p('padding', padding)
        self.p('output_padding', output_padding)
        self.p('data_format', data_format)
        self.p('activation', activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import Conv3DTransposeLayer
        from tfos.choices import ACTIVATIONS, BOOLEAN, PADDING

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        filters = param.get('filters')
        kernel_size = param.get('kernel_size')
        strides = param.get('strides', '1,1,1')
        padding = param.get('padding', PADDING[0])
        output_padding = param.get('output_padding', '')
        data_format = param.get('data_format', '')
        activation = param.get('activation', ACTIVATIONS[0])
        use_bias = param.get('use_bias', BOOLEAN[0])
        kernel_initializer = param.get('kernel_initializer', 'glorot_uniform')
        bias_initializer = param.get('bias_initializer', 'zeros')
        kernel_regularizer = param.get('kernel_regularizer', '')
        bias_regularizer = param.get('bias_regularizer', '')
        activity_regularizer = param.get('activity_regularizer', '')
        kernel_constraint = param.get('kernel_constraint', '')
        bias_constraint = param.get('bias_constraint', '')
        input_shape = param.get('input_shape')  # many integer separate with comma

        # 必传参数
        kernel_size = tuple([int(i) for i in kernel_size.split(',') if i])
        assert len(kernel_size) == 3, "Convolution3DTranspose kernel_size must be 3 dimension!"
        kwargs = dict(filters=int(filters), kernel_size=kernel_size)

        if strides:
            strides = tuple([int(i) for i in strides.split(',') if i])
            assert len(strides) == 3, "Convolution3DTranspose strides must be 3 dimension!"
            kwargs['strides'] = strides
        if padding:
            kwargs['padding'] = padding
        if output_padding:
            output_padding = tuple([int(i) for i in output_padding.split(',') if i])
            assert len(output_padding) == 3, "Convolution3DTranspose output_padding must be 3 dimension!"
            kwargs['output_padding'] = output_padding
        if data_format:
            kwargs['data_format'] = data_format
        if activation:
            kwargs['activation'] = activation
        if use_bias:
            kwargs['use_bias'] = True if use_bias.lower() == 'true' else False
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = Conv3DTransposeLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Conv2D', output_df)


class TestConvolutionSequence(unittest.TestCase):
    def tearDown(self) -> None:
        reset()

    # @unittest.skip('')
    def test_conv1d(self):
        Convolution1D(filters=32, kernel_size='3', input_shape='100,3').run()
        Convolution1D(filters=64, kernel_size='3').run()
        Convolution1D(filters=256, kernel_size='3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv2d(self):
        Convolution2D(filters=32, kernel_size='3,3', input_shape='100,100,3').run()
        Convolution2D(filters=64, kernel_size='3,3').run()
        Convolution2D(filters=256, kernel_size='3,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv3d(self):
        Convolution3D(filters=32, kernel_size='3,3,3', input_shape='100,100,100,3').run()
        Convolution3D(filters=64, kernel_size='3,3,3').run()
        Convolution3D(filters=256, kernel_size='3,3,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv2d_transpose(self):
        Convolution2DTranspose(filters=32, kernel_size='3,3', input_shape='100,100,3').run()
        Convolution2DTranspose(filters=64, kernel_size='3,3').run()
        Convolution2DTranspose(filters=256, kernel_size='3,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv3d_transpose(self):
        Convolution3DTranspose(filters=32, kernel_size='3,3,3', input_shape='100,100,100,3').run()
        Convolution3DTranspose(filters=64, kernel_size='3,3,3').run()
        Convolution3DTranspose(filters=256, kernel_size='3,3,3').run()
        SummaryLayer().run()


class TestConvolutionNetwork(unittest.TestCase):
    def tearDown(self) -> None:
        reset()

    # @unittest.skip('')
    def test_conv1d(self):
        InputLayer('100,3').run()
        Convolution1D(filters=32, kernel_size='3').run()
        Convolution1D(filters=64, kernel_size='3').run()
        Convolution1D(filters=256, kernel_size='3', strides='1').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv2d(self):
        InputLayer('100,100,3').run()
        Convolution2D(filters=32, kernel_size='3,3').run()
        Convolution2D(filters=64, kernel_size='3,3').run()
        Convolution2D(filters=256, kernel_size='3,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv3d(self):
        InputLayer('100,100,100,3').run()
        Convolution3D(filters=32, kernel_size='3,3,3').run()
        Convolution3D(filters=64, kernel_size='3,3,3').run()
        Convolution3D(filters=256, kernel_size='3,3,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv2d_transpose(self):
        InputLayer('100,100,3').run()
        Convolution2DTranspose(filters=32, kernel_size='3,3').run()
        Convolution2DTranspose(filters=64, kernel_size='3,3').run()
        Convolution2DTranspose(filters=256, kernel_size='3,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_conv3d_transpose(self):
        InputLayer('100,100,100,3').run()
        Convolution3DTranspose(filters=32, kernel_size='3,3,3').run()
        Convolution3DTranspose(filters=64, kernel_size='3,3,3').run()
        Convolution3DTranspose(filters=256, kernel_size='3,3,3').run()
        SummaryLayer().run()


if __name__ == "__main__":
    unittest.main()
