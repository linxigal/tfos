#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/11 12:36
:File   : config.py
"""

# keras.activations
valid_activations = [
    'softmax',
    'elu',
    'selu',
    'softplus',
    'softsign',
    'relu',
    'tanh',
    'sigmoid',
    'exponential',
    'hard_sigmoid',
    'linear',
]

# keras.losses
valid_losses = [
    'mse', 'MSE', 'mean_squared_error',
    'mae', 'MAE', 'mean_absolute_error',
    'mape', 'MAPE', 'mean_absolute_percentage_error',
    'msle', 'MSLE', 'mean_squared_logarithmic_error',
    'kld', 'KLD', 'kullback_leibler_divergence',  # 从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
    'cosine', 'cosine_proximity',  # 即预测值与真实标签的余弦距离平均值的相反数
    'squared_hinge',
    'hinge',
    'categorical_hinge',
    'logcosh',
    'categorical_crossentropy',  # 多类的对数损失, 注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
    'sparse_categorical_crossentropy',  # 如上，但接受稀疏标签, 需要在标签数据上增加一个维度：np.expand_dims(y,-1)
    'binary_crossentropy',  # 对数损失， log loss
    'poisson',  # 即(predictions - targets * log(predictions))的均值
]

# keras.metrics
valid_metrics = valid_losses + [
    'accuracy',  # 手动添加，keras代码中暂未找到
    'binary_accuracy',
    'categorical_accuracy',
    'sparse_categorical_accuracy',
    'top_k_categorical_accuracy',
    'sparse_top_k_categorical_accuracy',
]

# keras.optimizers
valid_optimizers = [
    'sgd', 'SGD',
    'rmsprop', 'RMSprop',
    'adagrad', 'Adagrad',
    'adadelta', 'Adadelta',
    'adam', 'Adam',
    'adamax', 'Adamax',
    'nadam', 'Nadam',
]

# keras.regularizers
valid_regularizers = [
    'l1', 'l2', 'l1_l2'
]
