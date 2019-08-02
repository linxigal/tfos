#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 12:45
:File       : drop.py
"""

from deep_insight.base import *


class Dropout(Base):
    def __init__(self, input_model_config_name, rate, noise_shape=None, seed=None):
        super(Dropout, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('rate', rate)
        self.p('noise_shape', noise_shape)
        self.p('seed', seed)

    def run(self):
        param = self.params

        from tfos.layers import DropoutLayer

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        rate = param.get('rate')
        noise_shape = param.get('noise_shape')
        seed = param.get('seed')
        model_rdd = inputRDD(input_model_config_name)
        outputdf = DropoutLayer(model_rdd, sc, sqlc).add(rate, noise_shape, seed)
        outputRDD('<#zzjzRddName#>_dropout', outputdf)


if __name__ == "__main__":
    from deep_insight.layers import Dense

    Dense(lrn(), 512, input_dim=784).run()
    Dropout(lrn(), 0.01).run()
    Dropout(lrn(), 0.01).run()
    Dropout(lrn(), 0.01).run()
    print_pretty()
