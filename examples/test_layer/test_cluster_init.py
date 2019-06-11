#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 14:32
:File       : test_cluster_init.py
"""

from examples.base import *


class ClusterInit(Base):
    def __init__(self, cluster_size, num_ps):
        super(ClusterInit, self).__init__()
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)

    def run(self):
        param = self.params

        import json
        from pyspark.sql import Row

        # param = json.loads('<#zzjzParam#>')
        cluster_size = param.get('cluster_size')
        num_ps = param.get('num_ps')

        outputdf = sqlc.createDataFrame([Row(cluster_params=json.dumps(param))])
        outputRDD('<#zzjzRddName#>_init', outputdf)
