#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time   : 2019/6/17 10:37
:File   : read_mnist.py
"""

from deep_insight.base import *


class ReadMnist(Base):
    def __init__(self, input_path, format='tfr'):
        super(ReadMnist, self).__init__()
        self.p('input_path', [{"path": input_path}])
        self.p('format', format)

    def run(self):
        param = self.params

        from tfos.data import DataSet

        # param = json.loads('<#zzjzParam#>')
        data_format = param.get('format')
        input_path = param.get('input_path')[0]['path']
        output_df = DataSet(sc).read_data(input_path, data_format=data_format)
        outputRDD('<#zzjzRddName#>_mnist', output_df)


if __name__ == "__main__":
    output_data_name = "<#zzjzRddName#>_mnist_tfr"
    input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/train"
    ReadMnist(input_path, 'tfr').run()
