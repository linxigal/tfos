#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/12 16:39
:File       : test_read_csv.py
"""

import logging
from examples.base import *


class TestReadCsv(Base):

    def __init__(self, filepath):
        super(TestReadCsv, self).__init__()
        self.p('filepath', filepath)

    def run(self):
        param = self.params

        # param = json.loads('<#zzjzParam#>')
        filepath = param.get('filepath')
        # rdd = sc.textFile(filepath)
        df = sqlc.read.csv(filepath, header=True)
        # rdd = rdd.map(lambda x: x.split(','))
        # df = rdd.toDF(['field_{}'.format(i) for i in range(len(rdd.first()))])
        outputRDD('<#zzjzRddName#>_data', df.rdd)


if __name__ == "__main__":
    import os
    from examples import ROOT_PATH

    filepath = os.path.join(ROOT_PATH, 'output_data', 'data', 'regression_data.csv')
    TestReadCsv(filepath).run()
