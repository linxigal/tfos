#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:46
:File   : optimizer.py
"""

from deep_insight.base import *


class Optimizer(Base):
    def __init__(self, input_model_config_name, loss, optimizer, metrics=None):
        super(Optimizer, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('loss', loss)
        self.p('optimizer', optimizer)
        self.p('metrics', metrics)

    def run(self):
        param = self.params

        from tfos.optimizers import OptimizerLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        loss = param.get("loss")
        optimizer = param.get('optimizer')
        metrics = param.get('metrics')

        model_rdd = inputRDD(input_model_config_name)
        outputdf = OptimizerLayer(model_rdd, sc, sqlc).add(loss, optimizer, metrics)
        outputRDD('<#zzjzRddName#>_optimizer', outputdf)


if __name__ == "__main__":
    from deep_insight.layers import Dense

    Dense(lrn(), 512, input_dim=784).run()
    Optimizer(lrn(), 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()
    inputRDD(lrn()).show()
    print_pretty()
