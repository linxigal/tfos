#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/19 15:13
:File   : core.py
"""

import unittest

from deep_insight.base import *
from deep_insight.layers.input import InputLayer


class Masking(Base):
    """覆盖层
    使用覆盖值覆盖序列，以跳过时间步。

    对于输入张量的每一个时间步（张量的第一个维度）， 如果所有时间步中输入张量的值与 mask_value 相等，
    那么这个时间步将在所有下游层被覆盖 (跳过) （只要它们支持覆盖）。

    如果任何下游层不支持覆盖但仍然收到此类输入覆盖信息，会引发异常。

    例如：

    考虑将要喂入一个 LSTM 层的 Numpy 矩阵 x， 尺寸为 (samples, timesteps, features)。
    你想要覆盖时间步 #3 和 #5，因为你缺乏这几个 时间步的数据。你可以：

    设置 x[:, 3, :] = 0. 以及 x[:, 5, :] = 0.
    在 LSTM 层之前，插入一个 mask_value=0 的 Masking 层：

        - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
        - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(LSTM(32))
    ```

    参数：
        mask_value: 覆盖值
            对于指定的列，使用该值进行覆盖
    """

    def __init__(self, mask_value='0', input_shape=None):
        super(Masking, self).__init__()
        self.p('mask_value', mask_value)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params
        from tfos.layers import MaskingLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        mask_value = param.get("mask_value", '0')
        input_shape = param.get("input_shape", '')

        kwargs = {}
        if mask_value:
            kwargs['mask_value'] = int(mask_value)
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = MaskingLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Masking', output_df)


class Dropout(Base):
    """Dropout层
    将 Dropout 应用于输入。

    Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。

    参数：
        rate: 丢弃率
            在 0 和 1 之间浮动。需要丢弃的输入比例。
        noise_shape: 噪声空间
            1D 整数张量， 表示将与输入相乘的二进制 dropout 掩层的形状。 例如，如果你的输入尺寸为 (batch_size, timesteps,
            features)，然后 你希望 dropout 掩层在所有时间步都是一样的， 你可以使用 noise_shape=(batch_size, 1, features)。
        seed: 随机种子
            一个作为随机种子的 Python 整数。

    """

    def __init__(self, rate, noise_shape=None, seed=None):
        super(Dropout, self).__init__()
        self.p('rate', rate)
        self.p('noise_shape', noise_shape)
        self.p('seed', seed)

    def run(self):
        param = self.params
        from tfos.layers import DropoutLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        rate = param.get("rate")
        noise_shape = param.get("noise_shape", "")
        seed = param.get("seed", "")

        # 必填参数
        kwargs = dict(rate=float(rate))
        # 可选参数
        if noise_shape:
            noise_shape = tuple([int(i) for i in noise_shape.split(',') if i])
            kwargs['noise_shape'] = noise_shape
        if seed:
            kwargs['seed'] = int(seed)

        model_rdd = inputRDD(input_prev_layers)
        output_df = DropoutLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Dropout', output_df)


class SpatialDropout1D(Base):
    """SpatialDropout1D层
    Dropout 的 Spatial 1D 版本

    此版本的功能与 Dropout 相同，但它会丢弃整个 1D 的特征图而不是丢弃单个元素。如果特征图中相邻的帧是强相关的
    （通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，
    SpatialDropout1D 将有助于提高特征图之间的独立性，应该使用它来代替 Dropout。

    输入尺寸
        3D 张量，尺寸为：(samples, timesteps, channels)

    输出尺寸
        与输入相同。

    参数：
        rate: 丢弃率
            0 到 1 之间的浮点数。需要丢弃的输入比例。
    """

    def __init__(self, rate):
        super(SpatialDropout1D, self).__init__()
        self.p('rate', rate)

    def run(self):
        param = self.params

        from tfos.layers import SpatialDropout1DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        rate = param.get("rate")

        # 必填参数
        kwargs = dict(rate=float(rate))

        model_rdd = inputRDD(input_prev_layers)
        output_df = SpatialDropout1DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_SpatialDropout1D', output_df)


class SpatialDropout2D(Base):
    """SpatialDropout2D层
    Dropout 的 Spatial 2D 版本

    此版本的功能与 Dropout 相同，但它会丢弃整个 2D 的特征图而不是丢弃单个元素。如果特征图中相邻的像素是强相关的
    （通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，
    SpatialDropout2D 将有助于提高特征图之间的独立性，应该使用它来代替 dropout。

    输入尺寸
        4D 张量，如果 data_format＝channels_first，尺寸为 (samples, channels, rows, cols)，
        如果 data_format＝channels_last，尺寸为 (samples, rows, cols, channels)

    输出尺寸
        与输入相同。

    参数：
    rate：丢弃率
        0 到 1 之间的浮点数。需要丢弃的输入比例。
    data_format：数据格式
        channels_first 或者 channels_last。在 channels_first 模式中，通道维度（即深度）位于索引 1，在 channels_last
        模式中，通道维度位于索引 3。默认为 image_data_format 的值，你可以在 Keras 的配置文件 ~/.keras/keras.json 中
        找到它。如果你从未设置过它，那么它将是 channels_last

    """

    def __init__(self, rate, data_format=None):
        super(SpatialDropout2D, self).__init__()
        self.p('rate', rate)
        self.p('data_format', data_format)

    def run(self):
        param = self.params
        from tfos.layers import SpatialDropout2DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        rate = param.get("rate")
        data_format = param.get("data_format", "")

        # 必填参数
        kwargs = dict(rate=float(rate))
        # 选填参数
        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = SpatialDropout2DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_SpatialDropout2D', output_df)


class SpatialDropout3D(Base):
    """SpatialDropout3D层
    Dropout 的 Spatial 3D 版本

    此版本的功能与 Dropout 相同，但它会丢弃整个 3D 的特征图而不是丢弃单个元素。如果特征图中相邻的体素是强相关的
    （通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，
    SpatialDropout3D 将有助于提高特征图之间的独立性，应该使用它来代替 dropout。

    输入尺寸
        5D 张量，如果 data_format＝channels_first，尺寸为 (samples, channels, dim1, dim2, dim3)，如果
        data_format＝channels_last，尺寸为 (samples, dim1, dim2, dim3, channels)

    输出尺寸
        与输入相同。

    参数：
    rate：丢弃率
        0 到 1 之间的浮点数。需要丢弃的输入比例。
    data_format：数据格式
        channels_first 或者 channels_last。在 channels_first 模式中，通道维度（即深度）位于索引 1，在 channels_last
        模式中，通道维度位于索引 4。默认为 image_data_format 的值，你可以在 Keras 的配置文件 ~/.keras/keras.json
        中找到它。如果你从未设置过它，那么它将是 channels_last
    """

    def __init__(self, rate, data_format=None):
        super(SpatialDropout3D, self).__init__()
        self.p('rate', rate)
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import SpatialDropout3DLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        rate = param.get("rate")
        data_format = param.get("data_format", "")

        # 必填参数
        kwargs = dict(rate=float(rate))
        # 选填参数
        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = SpatialDropout3DLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_SpatialDropout3D', output_df)


class Activation(Base):
    """激活层
    将激活函数应用于输出

    输入尺寸
        任意尺寸。 当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。

    输出尺寸
        与输入相同。

    参数：
        activation: 激活函数
            要使用的激活函数的名称 (详见: activations)， 或者选择一个 Theano 或 TensorFlow 操作。

    """

    def __init__(self, activation):
        super(Activation, self).__init__()
        self.p('activation', activation)

    def run(self):
        param = self.params

        from tfos.layers import ActivationLayer
        from tfos.choices import ACTIVATIONS

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        activation = param.get("activation", ACTIVATIONS[0])

        # 必填参数
        kwargs = dict(activation=activation)

        model_rdd = inputRDD(input_prev_layers)
        output_df = ActivationLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Activation', output_df)


class Reshape(Base):
    """Reshape层
    将输入重新调整为特定的尺寸。

    参数：
        target_shape: 目标尺寸
            整数元组。 不包含表示批量的轴。

    输入尺寸
        任意，尽管输入尺寸中的所有维度必须是固定的。 当使用此层作为模型中的第一层时， 使用参数 `input_shape `
        （整数元组，不包括样本数的轴）。

    输出尺寸
        `(batch_size,) + target_shape`

    例如：

    ```python
        # 作为 Sequential 模型的第一层
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # 现在：model.output_shape == (None, 3, 4)
        # 注意： `None` 是批表示的维度

        # 作为 Sequential 模型的中间层
        model.add(Reshape((6, 2)))
        # 现在： model.output_shape == (None, 6, 2)

        # 还支持使用 `-1` 表示维度的尺寸推断
        model.add(Reshape((-1, 2, 2)))
        # 现在： model.output_shape == (None, 3, 2, 2)
    ```
    """

    def __init__(self, target_shape, input_shape=None):
        super(Reshape, self).__init__()
        self.p('target_shape', target_shape)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params
        from tfos.layers import ReshapeLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        target_shape = param.get("target_shape")
        input_shape = param.get("input_shape")

        # 必填参数
        kwargs = dict(target_shape=tuple([int(i) for i in target_shape.split(',') if i]))
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = ReshapeLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Reshape', output_df)


class Permute(Base):
    """Permute层
    根据给定的模式置换输入的维度。

    在某些场景下很有用，例如将 RNN 和 CNN 连接在一起。

    例如：

    ```python
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        # 现在： model.output_shape == (None, 64, 10)
        # 注意： `None` 是批表示的维度
    ```

    参数：
        dims: 置换模式
            整数元组，置换模式，不包含样本维度。 索引从 1 开始。 例如, `(2, 1)` 置换输入的第一和第二个维度。

    输入尺寸
        任意。当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。

    输出尺寸
        与输入尺寸相同，但是维度根据指定的模式重新排列。

    """

    def __init__(self, dims, input_shape=None):
        super(Permute, self).__init__()
        self.p('dims', dims)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import PermuteLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        dims = param.get("dims")
        input_shape = param.get("input_shape")

        # 必填参数
        kwargs = dict(dims=tuple([int(i) for i in dims.split(',') if i]))
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        print(kwargs)
        model_rdd = inputRDD(input_prev_layers)
        output_df = PermuteLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Dropout', output_df)


class Flatten(Base):
    """Flatten层
    将输入展平。不影响批量大小。

    参数：
        data_format：数据格式
            一个字符串，其值为 `channels_last`（默认值）或者 `channels_first`。它表明输入的维度的顺序。此参数的目的是当
            模型从一种数据格式切换到另一种数据格式时保留权重顺序。`channels_last` 对应着尺寸为 `(batch, ..., channels)`
            的输入，而 `channels_first` 对应着尺寸为 `(batch, channels, ...)` 的输入。默认为 `image_data_format` 的值，
            你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

    例如：

    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),input_shape=(3, 32, 32), padding='same',))
        # 现在：model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # 现在：model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None):
        super(Flatten, self).__init__()
        self.p('data_format', data_format)

    def run(self):
        param = self.params

        from tfos.layers import FlattenLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        data_format = param.get("data_format", "")

        # 参数
        kwargs = dict()
        # 可选参数
        if data_format:
            kwargs['data_format'] = data_format

        model_rdd = inputRDD(input_prev_layers)
        output_df = FlattenLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Flatten', output_df)


class RepeatVector(Base):
    """RepeatVector层
    将输入重复 n 次。

    例如：

    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # 现在： model.output_shape == (None, 32)
        # 注意： `None` 是批表示的维度

        model.add(RepeatVector(3))
        # 现在： model.output_shape == (None, 3, 32)
    ```

    参数：
        n: 重复次数
            整数，重复次数

    输入尺寸
        2D 张量，尺寸为 (num_samples, features)。

    输出尺寸
        3D 张量，尺寸为 (num_samples, n, features)。
    """

    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.p('n', n)

    def run(self):
        param = self.params

        from tfos.layers import RepeatVectorLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        n = param.get("n")

        # 必填参数
        kwargs = dict(n=int(n))

        model_rdd = inputRDD(input_prev_layers)
        output_df = RepeatVectorLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_RepeatVector', output_df)


class Lambda(Base):
    """Lambda层
    将任意表达式封装为 Layer 对象。

    例如：

    ```python
        # 添加一个 x -> x^2 层
        model.add(Lambda(lambda x: x ** 2))
    ```

    ```python
        # 添加一个网络层，返回输入的正数部分与负数部分的反面的连接

        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)

        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        model.add(Lambda(antirectifier,output_shape=antirectifier_output_shape))
    ```

    参数：
        function: 函数
            需要封装的函数。 将输入张量作为第一个参数。
        output_shape: 输出尺寸
            预期的函数输出尺寸。 只在使用 Theano 时有意义。 可以是元组或者函数。 如果是元组，它只指定第一个维度；
            样本维度假设与输入相同：  `output_shape = (input_shape[0], ) + output_shape` 或者，输入是 `None` 且
            样本维度也是 `None`：  `output_shape = (None, ) + output_shape` 如果是函数，它指定整个尺寸为输入尺寸
            的一个函数：  `output_shape = f(input_shape)`
        mask: 掩膜
        arguments: 可选参数
            可选的需要传递给函数的关键字参数。

    输入尺寸
        任意。当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。

    输出尺寸
        由 output_shape 参数指定 (或者在使用 TensorFlow 时，自动推理得到)。
    """

    def __init__(self, function, output_shape=None, mask=None, arguments=None, input_shape=None):
        super(Lambda, self).__init__()
        self.p('function', function)
        self.p('output_shape', output_shape)
        self.p('mask', mask)
        self.p('arguments', arguments)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import LambdaLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        function = param.get("function")
        output_shape = param.get("output_shape", "")
        mask = param.get("mask", "")
        arguments = param.get("arguments", "")
        input_shape = param.get("input_shape", "")

        # 必填参数
        kwargs = dict(function=function)
        if output_shape:
            if callable(output_shape):
                kwargs['output_shape'] = output_shape
            else:
                kwargs['output_shape'] = tuple([int(i) for i in output_shape.split(',') if i])
        if mask:
            kwargs['mask'] = mask
        if arguments:
            kwargs['arguments'] = arguments
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = LambdaLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Lambda', output_df)


class Dense(Base):
    """全连接层
    `Dense` 实现以下操作： `output = activation(dot(input, kernel) + bias)` 其中 `activation` 是按逐个元素计算的激活函数，
    `kernel` 是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置向量 (只在 `use_bias` 为 `True` 时才有用)。

    注意: 如果该层的输入的秩大于2，那么它首先被展平然后 再计算与 `kernel` 的点乘。

    例如：

    ```python
        # 作为 Sequential 模型的第一层
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # 现在模型就会以尺寸为 (*, 16) 的数组作为输入，
        # 其输出数组的尺寸为 (*, 32)

        # 在第一层之后，你就不再需要指定输入的尺寸了：
        model.add(Dense(32))
    ```

    参数：
        units: 输出空间维度
            正整数，输出空间维度
        activation: 激活函数
            激活函数， 若不指定，则不使用激活函数 (即，「线性」激活: a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer:
            kernel 权值矩阵的初始化器
        bias_initializer: 偏置初始化
            偏置向量的初始化器
        kernel_regularizer: 权值正则化
            运用到 kernel 权值矩阵的正则化函数
        bias_regularizer: 偏置正则化
            运用到偏置向的的正则化函数
        activity_regularizer: 输出正则化
            运用到输出层的正则化函数
        kernel_constraint: 内核约束函数
            运用到 kernel 权值矩阵的约束函数
        bias_constraint: 偏置约束函数
            运用到偏置向量的约束函数
    """

    def __init__(self, units,
                 activation=None,
                 use_bias='True',
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None):
        super(Dense, self).__init__()
        self.p('units', units)
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

        from tfos.layers import DenseLayer
        from tfos.choices import BOOLEAN
        from tfos.choices import D_ACTIVATIONS

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        units = param.get("units")
        activation = param.get("activation", D_ACTIVATIONS[0])
        use_bias = param.get("use_bias", BOOLEAN[0])
        kernel_initializer = param.get("kernel_initializer", "glorot_uniform")
        bias_initializer = param.get("bias_initializer", "zeros")
        kernel_regularizer = param.get("kernel_regularizer", "")
        bias_regularizer = param.get("bias_regularizer", "")
        activity_regularizer = param.get("activity_regularizer", "")
        kernel_constraint = param.get("kernel_constraint", "")
        bias_constraint = param.get("bias_constraint", "")
        input_shape = param.get("input_shape", "")

        # 必填参数
        kwargs = dict(units=int(units))
        # 可选参数
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
        output_df = DenseLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Dense', output_df)


class ActivityRegularization(Base):
    """
    网络层，对基于代价函数的输入活动应用一个更新。

    参数:
        l1: L1 正则化因子
            (正数浮点型)。
        l2: L2 正则化因子
            (正数浮点型)。

    输入尺寸
        任意。当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。

    输出尺寸
        与输入相同。
    """

    def __init__(self, l1='0', l2='0', input_shape=None):
        super(ActivityRegularization, self).__init__()
        self.p('l1', l1)
        self.p('l2', l2)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.layers import ActivityRegularizationLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        l1 = param.get("l1", "0.")
        l2 = param.get("l2", "0.")
        input_shape = param.get("input_shape", "")

        # 参数
        kwargs = dict(l1=float(l1), l2=float(l2))
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = ActivityRegularizationLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_ActivityRegularization', output_df)


class TestCoreSequence(unittest.TestCase):

    def setUp(self) -> None:
        reset()

    # @unittest.skip('')
    def test_masking(self):
        Masking(input_shape='2,10').run()
        Masking().run()
        Masking().run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_dropout(self):
        Dense('512', input_shape='784').run()
        Dropout('0.01').run()
        Dropout('0.01').run()
        Dropout('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_spatial_dropout_1d(self):
        Dense('512', input_shape='32,784').run()
        SpatialDropout1D('0.01').run()
        SpatialDropout1D('0.01').run()
        SpatialDropout1D('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_spatial_dropout_2d(self):
        Dense('512', input_shape='32,64,784').run()
        SpatialDropout2D('0.01').run()
        SpatialDropout2D('0.01').run()
        SpatialDropout2D('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_spatial_dropout_3d(self):
        Dense('512', input_shape='32,64,128,784').run()
        SpatialDropout3D('0.01').run()
        SpatialDropout3D('0.01').run()
        SpatialDropout3D('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_activation(self):
        Dense('512', input_shape='784').run()
        Activation(activation='relu').run()
        Activation(activation='relu').run()
        Activation(activation='relu').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_reshape(self):
        Reshape('4,6', input_shape='24').run()
        Reshape('3,8').run()
        Reshape('-1,3,4').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_permute(self):
        Permute('2,1,3', '10,64,128').run()
        Permute('1,3,2').run()
        Permute('2,1,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_flatten(self):
        Dense('32', input_shape='10,64,128').run()
        Flatten().run()
        Flatten().run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_repeat_vector(self):
        Dense('32', input_shape='64').run()
        RepeatVector('3').run()
        SummaryLayer().run()

    @unittest.skip('')
    def test_lambda(self):
        from tensorflow.python.keras import backend as K

        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)

        Dense('32', input_shape='10,64').run()
        Lambda(antirectifier).run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_dense(self):
        Dense('512', input_shape='784').run()
        Dense('512').run()
        Dense('256').run()
        Dense('10').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_activity_regularization(self):
        # l1,l2不能同时为0
        ActivityRegularization('0.1', '0.5', input_shape='10,64').run()
        ActivityRegularization('0.6', '0.2').run()
        SummaryLayer().run()


class TestCoreNetWork(unittest.TestCase):

    def tearDown(self) -> None:
        reset()

    # @unittest.skip('')
    def test_masking(self):
        InputLayer('2,10').run()
        Masking().run()
        Masking().run()
        Masking().run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_dropout(self):
        InputLayer('784').run()
        Dense('512').run()
        Dropout('0.01').run()
        Dropout('0.01').run()
        Dropout('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_spatial_dropout_1d(self):
        InputLayer('32,784').run()
        Dense('512').run()
        SpatialDropout1D('0.01').run()
        SpatialDropout1D('0.01').run()
        SpatialDropout1D('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_spatial_dropout_2d(self):
        InputLayer('32,64,784').run()
        Dense('512').run()
        SpatialDropout2D('0.01').run()
        SpatialDropout2D('0.01').run()
        SpatialDropout2D('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_spatial_dropout_3d(self):
        InputLayer('32,64,128,784').run()
        Dense('512').run()
        SpatialDropout3D('0.01').run()
        SpatialDropout3D('0.01').run()
        SpatialDropout3D('0.01').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_activation(self):
        InputLayer('784').run()
        Dense('512').run()
        Activation(activation='relu').run()
        Activation(activation='relu').run()
        Activation(activation='relu').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_reshape(self):
        InputLayer('24').run()
        Reshape('4,6').run()
        Reshape('3,8').run()
        Reshape('-1,3,4').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_permute(self):
        InputLayer('10,64,128').run()
        Permute('2,1,3').run()
        Permute('1,3,2').run()
        Permute('2,1,3').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_flatten(self):
        InputLayer('10,64,128').run()
        Dense('32').run()
        Flatten().run()
        Flatten().run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_repeat_vector(self):
        InputLayer('64').run()
        Dense('32').run()
        RepeatVector('3').run()
        SummaryLayer().run()

    @unittest.skip('')
    def test_lambda(self):
        from tensorflow.python.keras import backend as K

        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)

        InputLayer('10,64').run()
        Dense('32').run()
        Lambda(antirectifier).run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_dense(self):
        InputLayer('784').run()
        Dense('512').run()
        Dense('256').run()
        Dense('10').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_activity_regularization(self):
        # l1,l2不能同时为0
        InputLayer('10,64').run()
        ActivityRegularization('0.1', '0.5').run()
        ActivityRegularization('0.6', '0.2').run()
        SummaryLayer().run()


if __name__ == '__main__':
    unittest.main()
