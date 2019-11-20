# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

from tfos.base.config import *

BOOLEAN = ['true', 'false']
PADDING = ['valid', 'same']
OUTPUT_FORMAT = ['json', 'csv']
MNIST_FORMAT = ['tfr', 'csv', 'pickle', 'gz', 'npz']
DATA_MODE = ['train', 'test']
OPERATORS = ['start', 'stop']
ACTIVATIONS = [
    'relu',
    'softmax',
    'tanh',
    'sigmoid',
    'elu',
    'selu',
    'softplus',
    'softsign',
    'exponential',
    'hard_sigmoid',
    'linear'
]
D_ACTIVATIONS = ['不使用'] + ACTIVATIONS

LOSSES = [
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'mean_squared_logarithmic_error',
    'kullback_leibler_divergence',  # 从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
    'cosine_proximity',  # 即预测值与真实标签的余弦距离平均值的相反数
    'squared_hinge',
    'hinge',
    'categorical_hinge',
    'logcosh',
    'categorical_crossentropy',  # 多类的对数损失, 注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
    'sparse_categorical_crossentropy',  # 如上，但接受稀疏标签, 需要在标签数据上增加一个维度：np.expand_dims(y,-1)
    'binary_crossentropy',  # 对数损失， log loss
    'poisson',  # 即(predictions - targets * log(predictions))的均值
]

METRICS = [
    'accuracy',  # 手动添加，keras代码中暂未找到
    'crossentropy',
    'binary_accuracy',
    'categorical_accuracy',
    'sparse_categorical_accuracy',
    'top_k_categorical_accuracy',
    'sparse_top_k_categorical_accuracy',
]
METRICS.extend(LOSSES)

OPTIMIZERS = [
    'sgd',
    'rmsprop',
    'adagrad',
    'adadelta',
    'adam',
    'adamax',
    'nadam',
]

REGULARIZES = valid_regularizers

# 单选框
SINGLE_BOX = {
    'padding': [
        ('true', 'true'),
        ('false', 'false')
    ]
}

# 复选框
CHECK_BOX = {

}

# 下拉菜单
DROP_DOWN_MENU = {
    'activation': [
        ('', "不使用"),
        ('softmax', 'softmax'),
        ('elu', 'elu'),
        ('selu', 'selu'),
        ('softplus', 'softplus'),
        ('softsign', 'softsign'),
        ('relu', 'relu'),
        ('tanh', 'tanh'),
        ('sigmoid', 'sigmoid'),
        ('exponential', 'exponential'),
        ('hard_sigmoid', 'hard_sigmoid'),
        ('linear', 'linear'),
    ]
}

# 图片格式

IMG_FORMAT = ['jpg', 'png', 'gif']
