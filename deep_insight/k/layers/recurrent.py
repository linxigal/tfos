#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/10 14:22
:File   :recurrent.py
:content:
  
"""
from deep_insight.base import *
from deep_insight.k.layers.input import InputLayer


class SimpleRNN(Base):
    """全连接的RNN
    全连接的 RNN，其输出将被反馈到输入。

    参数
        units: 输出空间
            正整数，输出空间的维度。
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 默认：双曲正切（tanh）。 如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 线性层权值初始化
            kernel 权值矩阵的初始化器， 用于输入的线性转换 (详见 initializers)。
        recurrent_initializer: 循环层权值初始化
            recurrent_kernel 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见initializers).
        kernel_regularizer: 线性层权值正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        recurrent_regularizer: 循环层权值正则化
            运用到 recurrent_kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 输出正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 线性层权值约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        recurrent_constraint: 循环层权值约束
            运用到 recurrent_kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。
        dropout: 单元丢弃比例
            在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
        recurrent_dropout: 循环层单元丢弃比例
            在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
        return_sequences: 返回全部序列
            布尔值。是返回输出序列中的最后一个输出，还是全部序列。
        return_state: 返回最后一个状态
            布尔值。除了输出之外是否返回最后一个状态。
        go_backwards: 向后处理序列
            布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
        stateful: 状态向下作用
            布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品 的最后状态将用作下一批次中索引 i 样品的初始状态。
        unroll: 网络展开
            布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。
            展开只适用于短序列。
    """

    def __init__(self,
                 units,
                 activation='tanh',
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout='0.',
                 recurrent_dropout='0.',
                 return_sequences='False',
                 return_state='False',
                 go_backwards='False',
                 stateful='False',
                 unroll='False',
                 input_shape=None):
        super(SimpleRNN, self).__init__()
        self.p('units', units)
        self.p('activation', activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('recurrent_initializer', recurrent_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('recurrent_regularizer', recurrent_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('recurrent_constraint', recurrent_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('dropout', dropout)
        self.p('recurrent_dropout', recurrent_dropout)
        self.p('return_sequences', return_sequences)
        self.p('return_state', return_state)
        self.p('go_backwards', go_backwards)
        self.p('stateful', stateful)
        self.p('unroll', unroll)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.k.layers import SimpleRNNLayer
        from tfos.choices import BOOLEAN, ACTIVATIONS
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers", '')
        units = param.get("units")
        activation = param.get("activation", ACTIVATIONS[2])
        use_bias = param.get("use_bias", BOOLEAN[0])
        kernel_initializer = param.get("kernel_initializer", 'glorot_uniform')
        recurrent_initializer = param.get("recurrent_initializer", 'orthogonal')
        bias_initializer = param.get("bias_initializer", 'zeros')
        kernel_regularizer = param.get("kernel_regularizer", '')
        recurrent_regularizer = param.get("recurrent_regularizer", '')
        bias_regularizer = param.get("bias_regularizer", '')
        activity_regularizer = param.get("activity_regularizer", '')
        kernel_constraint = param.get("kernel_constraint", '')
        recurrent_constraint = param.get("recurrent_constraint", '')
        bias_constraint = param.get("bias_constraint", '')
        dropout = param.get("dropout", '0')
        recurrent_dropout = param.get("recurrent_dropout", '0')
        return_sequences = param.get("return_sequences", BOOLEAN[1])
        return_state = param.get("return_state", BOOLEAN[1])
        go_backwards = param.get("go_backwards", BOOLEAN[1])
        stateful = param.get("stateful", BOOLEAN[1])
        unroll = param.get("unroll", BOOLEAN[1])
        input_shape = param.get("input_shape", '')

        # 必填参数
        kwargs = dict(units=int(units))
        # 选填参数
        kwargs['activation'] = activation
        kwargs['use_bias'] = use_bias
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if recurrent_initializer:
            kwargs['recurrent_initializer'] = recurrent_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if recurrent_regularizer:
            kwargs['recurrent_regularizer'] = recurrent_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if recurrent_constraint:
            kwargs['recurrent_constraint'] = recurrent_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if dropout:
            kwargs['dropout'] = float(dropout)
        if recurrent_dropout:
            kwargs['recurrent_dropout'] = float(recurrent_dropout)
        kwargs['return_sequences'] = convert_bool(return_sequences)
        kwargs['return_state'] = convert_bool(return_state)
        kwargs['go_backwards'] = convert_bool(go_backwards)
        kwargs['stateful'] = convert_bool(stateful)
        kwargs['unroll'] = convert_bool(unroll)
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = SimpleRNNLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Masking', output_df)


class GRU(Base):
    """
    门限循环单元网络（Gated Recurrent Unit） - Cho et al. 2014.

    有两种变体。默认的是基于 1406.1078v3 的实现，同时在矩阵乘法之前将复位门应用于隐藏状态。 另一种则是基于 1406.1078v1
    的实现，它包括顺序倒置的操作。

    第二种变体与 CuDNNGRU(GPU-only) 兼容并且允许在 CPU 上进行推理。 因此它对于 kernel 和 recurrent_kernel 有可分离偏置。
     使用 'reset_after'=True 和 recurrent_activation='sigmoid' 。

    参数
        units: 输出空间
            正整数，输出空间的维度。
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 默认：双曲正切 (tanh)。 如果传入 None，则不使用激活函数
             (即 线性激活：a(x) = x)。
        recurrent_activation: 循环层激活函数
            用于循环时间步的激活函数 (详见 activations)。 默认：分段线性近似 sigmoid (hard_sigmoid)。
            如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 线性层权值初始化
            kernel 权值矩阵的初始化器， 用于输入的线性转换 (详见 initializers)。
        recurrent_initializer: 循环层权值初始化
            recurrent_kernel 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见initializers).
        kernel_regularizer: 线性层权值正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        recurrent_regularizer: 循环层权值正则化
            运用到 recurrent_kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 输出正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 线性层权值约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        recurrent_constraint: 循环层权值约束
            运用到 recurrent_kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。
        dropout: 单元丢弃比例
            在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
        recurrent_dropout: 循环层单元丢弃比例
            在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
        implementation: 实现模式
            实现模式，1 或 2。 模式 1 将把它的操作结构化为更多的小的点积和加法操作， 而模式 2 将把它们分批到更少，更大的操作中。
            这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
        return_sequences: 返回全部序列
            布尔值。是返回输出序列中的最后一个输出，还是全部序列。
        return_state:  返回最后一个状态
            布尔值。除了输出之外是否返回最后一个状态。
        go_backwards: 向后处理序列
            布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
        stateful:  状态向下作用
            布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态。
        unroll: 网络展开
            布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。
             展开只适用于短序列。
        reset_after: 矩阵乘法之后使用重置门
            GRU 公约 (是否在矩阵乘法之前或者之后使用重置门)。 False =「之前」(默认)，Ture =「之后」( CuDNN 兼容)。
    """

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout='0.',
                 recurrent_dropout='0.',
                 implementation='1',
                 return_sequences='False',
                 return_state='False',
                 go_backwards='False',
                 stateful='False',
                 unroll='False',
                 reset_after='False',
                 input_shape=None):
        super(GRU, self).__init__()
        self.p('units', units)
        self.p('activation', activation)
        self.p('recurrent_activation', recurrent_activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('recurrent_initializer', recurrent_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('recurrent_regularizer', recurrent_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('recurrent_constraint', recurrent_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('dropout', dropout)
        self.p('recurrent_dropout', recurrent_dropout)
        self.p('implementation', implementation)
        self.p('return_sequences', return_sequences)
        self.p('return_state', return_state)
        self.p('go_backwards', go_backwards)
        self.p('stateful', stateful)
        self.p('unroll', unroll)
        self.p('reset_after', reset_after)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.k.layers import GRULayer
        from tfos.choices import BOOLEAN, ACTIVATIONS
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers", '')
        units = param.get("units")
        activation = param.get("activation", ACTIVATIONS[2])
        recurrent_activation = param.get("recurrent_activation", ACTIVATIONS[9])
        use_bias = param.get("use_bias", BOOLEAN[0])
        kernel_initializer = param.get("kernel_initializer", 'glorot_uniform')
        recurrent_initializer = param.get("recurrent_initializer", 'orthogonal')
        bias_initializer = param.get("bias_initializer", 'zeros')
        kernel_regularizer = param.get("kernel_regularizer", '')
        recurrent_regularizer = param.get("recurrent_regularizer", '')
        bias_regularizer = param.get("bias_regularizer", '')
        activity_regularizer = param.get("activity_regularizer", '')
        kernel_constraint = param.get("kernel_constraint", '')
        recurrent_constraint = param.get("recurrent_constraint", '')
        bias_constraint = param.get("bias_constraint", '')
        dropout = param.get("dropout", '0')
        recurrent_dropout = param.get("recurrent_dropout", '0')
        implementation = param.get("implementation", '1')
        return_sequences = param.get("return_sequences", BOOLEAN[1])
        return_state = param.get("return_state", BOOLEAN[1])
        go_backwards = param.get("go_backwards", BOOLEAN[1])
        stateful = param.get("stateful", BOOLEAN[1])
        unroll = param.get("unroll", BOOLEAN[1])
        reset_after = param.get("reset_after", BOOLEAN[1])
        input_shape = param.get("input_shape", '')

        # 必填参数
        kwargs = dict(units=int(units))
        # 选填参数
        if activation:
            kwargs['activation'] = activation
        if recurrent_activation:
            kwargs['recurrent_activation'] = recurrent_activation
        kwargs['use_bias'] = convert_bool(use_bias)
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if recurrent_initializer:
            kwargs['recurrent_initializer'] = recurrent_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if recurrent_regularizer:
            kwargs['recurrent_regularizer'] = recurrent_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if recurrent_constraint:
            kwargs['recurrent_constraint'] = recurrent_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if dropout:
            kwargs['dropout'] = float(dropout)
        if recurrent_dropout:
            kwargs['recurrent_dropout'] = float(recurrent_dropout)
        if implementation:
            kwargs['implementation'] = int(implementation)
        kwargs['return_sequences'] = convert_bool(return_sequences)
        kwargs['return_state'] = convert_bool(return_state)
        kwargs['go_backwards'] = convert_bool(go_backwards)
        kwargs['stateful'] = convert_bool(stateful)
        kwargs['unroll'] = convert_bool(unroll)
        kwargs['reset_after'] = convert_bool(reset_after)
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = GRULayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Masking', output_df)


class LSTM(Base):
    """长短期记忆网络层
    长短期记忆网络层（Long Short-Term Memory） - Hochreiter 1997.

    参数
        units: 输出空间
            正整数，输出空间的维度。
        activation: 激活函数
            要使用的激活函数 (详见 activations)。 如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
        recurrent_activation: 循环层激活函数
            用于循环时间步的激活函数 (详见 activations)。 默认：分段线性近似 sigmoid (hard_sigmoid)。
            如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
        use_bias: 使用偏置
            布尔值，该层是否使用偏置向量。
        kernel_initializer: 线性层权值初始化
            kernel 权值矩阵的初始化器， 用于输入的线性转换 (详见 initializers)。
        recurrent_initializer: 循环层权值初始化
            recurrent_kernel 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 initializers)。
        bias_initializer: 偏置初始化
            偏置向量的初始化器 (详见initializers).
        unit_forget_bias: 初始化时修改忘记门偏置
            布尔值。 如果为 True，初始化时，将忘记门的偏置加 1。 将其设置为 True 同时还会强制 bias_initializer="zeros"。
            这个建议来自 Jozefowicz et al.。
        kernel_regularizer: 线性层权值正则化
            运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        recurrent_regularizer: 循环层权值正则化
            运用到 recurrent_kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 偏置正则化
            运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 输出正则化
            运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 线性层权值约束
            运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        recurrent_constraint: 循环层权值约束
            运用到 recurrent_kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 偏置约束
            运用到偏置向量的约束函数 (详见 constraints)。
        dropout: 单元丢弃比例
            在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
        recurrent_dropout: 循环层单元丢弃比例
            在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
        implementation: 实现模式
            实现模式，1 或 2。 模式 1 将把它的操作结构化为更多的小的点积和加法操作， 而模式 2 将把它们分批到更少，更大的操作中。
            这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
        return_sequences: 返回全部序列
            布尔值。是返回输出序列中的最后一个输出，还是全部序列。
        return_state: 返回最后一个状态
            布尔值。除了输出之外是否返回最后一个状态。
        go_backwards: 向后处理序列
            布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
        stateful: 状态向下作用
            布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态。
        unroll: 网络展开
            布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。
             展开只适用于短序列。
    """

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias='True',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias='True',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout='0.',
                 recurrent_dropout='0.',
                 implementation='1',
                 return_sequences='False',
                 return_state='False',
                 go_backwards='False',
                 stateful='False',
                 unroll='False',
                 input_shape=None):
        super(LSTM, self).__init__()
        self.p('units', units)
        self.p('activation', activation)
        self.p('recurrent_activation', recurrent_activation)
        self.p('use_bias', use_bias)
        self.p('kernel_initializer', kernel_initializer)
        self.p('recurrent_initializer', recurrent_initializer)
        self.p('bias_initializer', bias_initializer)
        self.p('unit_forget_bias', unit_forget_bias)
        self.p('kernel_regularizer', kernel_regularizer)
        self.p('recurrent_regularizer', recurrent_regularizer)
        self.p('bias_regularizer', bias_regularizer)
        self.p('activity_regularizer', activity_regularizer)
        self.p('kernel_constraint', kernel_constraint)
        self.p('recurrent_constraint', recurrent_constraint)
        self.p('bias_constraint', bias_constraint)
        self.p('dropout', dropout)
        self.p('recurrent_dropout', recurrent_dropout)
        self.p('implementation', implementation)
        self.p('return_sequences', return_sequences)
        self.p('return_state', return_state)
        self.p('go_backwards', go_backwards)
        self.p('stateful', stateful)
        self.p('unroll', unroll)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        from tfos.k.layers import LSTMLayer
        from tfos.choices import BOOLEAN, ACTIVATIONS
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers", '')
        units = param.get("units")
        activation = param.get("activation", ACTIVATIONS[2])
        recurrent_activation = param.get("recurrent_activation", ACTIVATIONS[9])
        use_bias = param.get("use_bias", BOOLEAN[0])
        kernel_initializer = param.get("kernel_initializer", 'glorot_uniform')
        recurrent_initializer = param.get("recurrent_initializer", 'orthogonal')
        bias_initializer = param.get("bias_initializer", 'zeros')
        unit_forget_bias = param.get("unit_forget_bias", BOOLEAN[0])
        kernel_regularizer = param.get("kernel_regularizer", '')
        recurrent_regularizer = param.get("recurrent_regularizer", '')
        bias_regularizer = param.get("bias_regularizer", '')
        activity_regularizer = param.get("activity_regularizer", '')
        kernel_constraint = param.get("kernel_constraint", '')
        recurrent_constraint = param.get("recurrent_constraint", '')
        bias_constraint = param.get("bias_constraint", '')
        dropout = param.get("dropout", '0')
        recurrent_dropout = param.get("recurrent_dropout", '0')
        implementation = param.get("implementation", '1')
        return_sequences = param.get("return_sequences", BOOLEAN[1])
        return_state = param.get("return_state", BOOLEAN[1])
        go_backwards = param.get("go_backwards", BOOLEAN[1])
        stateful = param.get("stateful", BOOLEAN[1])
        unroll = param.get("unroll", BOOLEAN[1])
        input_shape = param.get("input_shape", '')

        # 必填参数
        kwargs = dict(units=int(units))
        # 选填参数
        if activation:
            kwargs['activation'] = activation
        if recurrent_activation:
            kwargs['recurrent_activation'] = recurrent_activation
        kwargs['use_bias'] = use_bias
        if kernel_initializer:
            kwargs['kernel_initializer'] = kernel_initializer
        if recurrent_initializer:
            kwargs['recurrent_initializer'] = recurrent_initializer
        if bias_initializer:
            kwargs['bias_initializer'] = bias_initializer
        if kernel_regularizer:
            kwargs['kernel_regularizer'] = kernel_regularizer
        if recurrent_regularizer:
            kwargs['recurrent_regularizer'] = recurrent_regularizer
        if bias_regularizer:
            kwargs['bias_regularizer'] = bias_regularizer
        kwargs['unit_forget_bias'] = convert_bool(unit_forget_bias)
        if activity_regularizer:
            kwargs['activity_regularizer'] = activity_regularizer
        if kernel_constraint:
            kwargs['kernel_constraint'] = kernel_constraint
        if recurrent_constraint:
            kwargs['recurrent_constraint'] = recurrent_constraint
        if bias_constraint:
            kwargs['bias_constraint'] = bias_constraint
        if dropout:
            kwargs['dropout'] = float(dropout)
        if recurrent_dropout:
            kwargs['recurrent_dropout'] = float(recurrent_dropout)
        if implementation:
            kwargs['implementation'] = int(implementation)
        kwargs['return_sequences'] = convert_bool(return_sequences)
        kwargs['return_state'] = convert_bool(return_state)
        kwargs['go_backwards'] = convert_bool(go_backwards)
        kwargs['stateful'] = convert_bool(stateful)
        kwargs['unroll'] = convert_bool(unroll)
        if input_shape:
            kwargs['input_shape'] = tuple([int(i) for i in input_shape.split(',') if i])

        model_rdd = inputRDD(input_prev_layers)
        output_df = LSTMLayer(model_rdd, sqlc=sqlc).add(**kwargs)
        outputRDD('<#zzjzRddName#>_Masking', output_df)


class TestRNN(unittest.TestCase):

    def tearDown(self) -> None:
        reset()

    # @unittest.skip("")
    def test_simple_rnn(self):
        SimpleRNN('32', input_shape='28,28').run()
        SummaryLayer().run()

    # @unittest.skip("")
    def test_gru(self):
        GRU('32', input_shape='28,28').run()
        SummaryLayer().run()

    # @unittest.skip("")
    def test_lstm(self):
        LSTM('32', input_shape='28,28').run()
        SummaryLayer().run()


class TestRNNNetwork(unittest.TestCase):

    def tearDown(self) -> None:
        reset()

    # @unittest.skip("")
    def test_simple_rnn(self):
        InputLayer('28,28').run()
        SimpleRNN('32').run()
        SummaryLayer().run()

    # @unittest.skip("")
    def test_gru(self):
        InputLayer('28,28').run()
        GRU('32').run()
        SummaryLayer().run()

    # @unittest.skip("")
    def test_lstm(self):
        InputLayer('28,28').run()
        LSTM('32').run()
        SummaryLayer().run()


if __name__ == '__main__':
    unittest.main()
