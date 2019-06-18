#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:46
:File   : test_optimizer.py
"""

from examples.base import *


class TestOptimizer(Base):
    def __init__(self, output_rdd_name, loss, optimizer, metrics=None):
        super(TestOptimizer, self).__init__()
        self.p('output_rdd_name', output_rdd_name)
        self.p('loss', loss)
        self.p('optimizer', optimizer)
        self.p('metrics', metrics)

    def run(self):
        param = self.params

        valid_loss = ['mean_squared_error', 'mse',
                      'mean_absolute_error', 'mae',
                      'mean_absolute_percentage', 'mape',
                      'mean_squared_logarithmic_error', 'msle',
                      'squared_hinge',
                      'hinge',
                      'categorical_hinge',
                      'binary_crossentropy',  # 对数损失， logloss
                      'logcosh',
                      'categorical_crossentropy',  # 多类的对数损失, 注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
                      'sparse_categorical_crossentropy',  # 如上，但接受稀疏标签, 需要在标签数据上增加一个维度：np.expand_dims(y,-1)
                      'kullback_leibler_divergence',  # 从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
                      'poisson',  # 即(predictions - targets * log(predictions))的均值
                      'cosine_proximity',  # 即预测值与真实标签的余弦距离平均值的相反数
                      ]
        valid_optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
        valid_metrics = ['accuracy']

        # param = json.loads('<#zzjzParam#>')
        output_rdd_name = param.get("output_rdd_name")
        loss = param.get("loss")
        optimizer = param.get('optimizer')
        metrics = param.get('metrics')

        if loss not in valid_loss:
            raise ValueError('model loss function incorrect!')
        if optimizer not in valid_optimizers:
            raise ValueError('model optimizer method incorrect!')

        check_metrics = []
        if metrics:
            if not isinstance(metrics, list):
                metrics = [metrics]
            for metric in metrics:
                if metric in valid_metrics:
                    check_metrics.append(metric)
                else:
                    raise ValueError(f"parameter metrics: {metric} is invalid!")

        optimizer_params = {
            'loss': loss,
            'optimizer': optimizer,
            'metrics': check_metrics if check_metrics else None
        }

        outputdf = dict2df(optimizer_params, 'compile_config')
        # outputRDD('<#zzjzRddName#>_optimizer', outputdf)
        outputRDD(output_rdd_name, outputdf)


if __name__ == "__main__":
    TestOptimizer('<#zzjzRddName#>_compile', 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()
    print_pretty('<#zzjzRddName#>_compile')
