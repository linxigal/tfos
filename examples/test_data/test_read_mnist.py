#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time   : 2019/6/17 10:37
:File   : test_read_mnist.py
"""

from examples.base import *


class TestReadMnist(Base):
    def __init__(self, input_path, format='tfr'):
        super(TestReadMnist, self).__init__()
        self.p('input_path', [{"path": input_path}])
        self.p('format', format)

    def run(self):
        param = self.params

        from tfos.data import DataSet

        # param = json.loads('<#zzjzParam#>')
        data_format = param.get('format')
        input_path = param.get('input_path')[0]['path']
        output_df = DataSet(sc).read_data(input_path, data_format=data_format)
        outputRDD('<#zzjzRddName#>_data', output_df)


if __name__ == "__main__":
    output_data_name = "<#zzjzRddName#>_mnist_tfr"
    input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/train"
    TestReadMnist(input_path, 'tfr').run()
