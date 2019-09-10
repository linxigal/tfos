#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:46
:File   : optimizer.py
"""

from deep_insight.base import *


class Optimizer(Base):
    def __init__(self, loss, optimizer, metrics=None):
        super(Optimizer, self).__init__()
        self.p('loss', loss)
        self.p('optimizer', optimizer)
        self.p('metrics', metrics)

    def run(self):
        param = self.params

        from tfos.optimizers import OptimizerLayer

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get("input_prev_layers")
        loss = param.get("loss")
        optimizer = param.get('optimizer')
        metrics = param.get('metrics')

        model_rdd = inputRDD(input_prev_layers)
        outputdf = OptimizerLayer(model_rdd, sc, sqlc).add(loss, optimizer, metrics)
        outputRDD('<#zzjzRddName#>_optimizer', outputdf)


if __name__ == "__main__":
    from deep_insight.layers import Dense

    Dense(512, input_dim=784).run()
    Optimizer('categorical_crossentropy', 'rmsprop', ['accuracy']).run()
    inputRDD(BRANCH).show()
    print_pretty()
