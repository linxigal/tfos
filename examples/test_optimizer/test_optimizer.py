#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:46
:File   : test_optimizer.py
"""

from examples.base import *


class TestOptimizer(Base):
    def __init__(self, output_rdd_name, loss, optimizer, metrics):
        super(TestOptimizer, self).__init__()
        self.p('output_rdd_name', output_rdd_name)
        self.p('loss', loss)
        self.p('optimizer', optimizer)
        self.p('metrics', metrics)

    def run(self):
        param = self.params

        valid_loss = ['mean_squared_error', 'binary_crossentropy', 'categorical_crossentropy', ]
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

        optimizer_params = {
            'loss': loss,
            'optimizer': optimizer,
            'metrics': check_metrics
        }

        outputdf = dict2df(optimizer_params, 'compile_config')
        # outputRDD('<#zzjzRddName#>_optimizer', outputdf)
        outputRDD(output_rdd_name, outputdf)


if __name__ == "__main__":
    TestOptimizer('<#zzjzRddName#>_compile', 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()
    print_pretty('<#zzjzRddName#>_compile')
