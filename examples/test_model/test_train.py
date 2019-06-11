#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 16:10
:File       : test_model_train.py
"""

from examples.base import *
from tensorflowonspark import TFCluster, TFNode


def main_fun(args, ctx):
    cluster, server = TFNode.start_cluster_server(ctx)

    if ctx.job_name == "ps":
        server.join()
    elif ctx.job_name == "worker":
        pass


class ModelTrain(Base):
    def __init__(self, input_table_name):
        super(ModelTrain, self).__init__()
        self.p('input_table_name', input_table_name)

    def run(self):
        param = self.params

        # param = json.loads('<#zzjzParam#>')
        input_table_name = param.get('input_table_name')

        input_rdd = inputRDD(input_table_name)

        cluster = TFCluster.run(main_fun)
        cluster.train(input_rdd)
