#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time   : 2019/6/17 10:37
:File   : read_mnist.py
"""

import unittest

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
        # output_df.show()
        outputRDD('<#zzjzRddName#>_mnist', output_df)


class TestReadMnist(unittest.TestCase):

    def test_read_mnist(self):
        input_path = "/home/wjl/github/tfos/data/mnist/tfr/test"
        ReadMnist(input_path, 'tfr').b(DATA_BRANCH).run()


if __name__ == "__main__":
    unittest.main()
