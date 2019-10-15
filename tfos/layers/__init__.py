#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 16:39
:File   : __init__.py.py
"""

from .activation import *
from .convolution import *
from .core import *
from .embeddings import *
from .input import *
from .merge import *
from .optimizer import *
from .pooling import *
from .recurrent import *
from .tensorboard import *
from .normalization import *

__all__ = [
    # input layer
    "InputLayer",
    # core layer
    "MaskingLayer", "DropoutLayer", "SpatialDropout1DLayer", "SpatialDropout2DLayer", "SpatialDropout3DLayer",
    "ActivationLayer", "ReshapeLayer", "PermuteLayer", "FlattenLayer", "RepeatVectorLayer", "LambdaLayer",
    "DenseLayer", "ActivityRegularizationLayer",
    # merge layer
    "AddL", "SubtractL", "MultiplyL", "AverageL", "MaximumL", "MinimumL", "ConcatenateL", "DotL",
    # convolution layer
    "Conv1DLayer", "Conv2DLayer", "Conv3DLayer", "Conv2DTransposeLayer", "Conv3DTransposeLayer",
    # activation layer
    "LeakyReLULayer", "PReLULayer", "ELULayer", "ThresholdedReLULayer", "SoftmaxLayer", "ReLULayer",
    # pooling layer
    "MaxPool1DLayer", "MaxPool2DLayer", "MaxPool3DLayer", "AvgPool1DLayer", "AvgPool2DLayer", "AvgPool3DLayer",
    # rnn layer
    "SimpleRNN", "GRU", "LSTM",
    # optimizer layer
    'SGDLayer', 'RMSpropLayer', 'AdagradLayer', 'AdadeltaLayer', 'AdamLayer', 'AdamaxLayer', 'NadamLayer',
    'TensorBoardLayer',
    'EmbeddingLayer',
    'BatchNormalizationLayer',
]
