#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/27 15:24
:File   :cifar10.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class Cifar10(Base):
    """cifar10数据集

    读取cifar-10数据集算子

    参数：
        data_dir: 数据目录
            cifar-10数据存放的目录
        mode: 读取模式
            读取数据的模式，分为训练数据和测试数据， train|test
        one_hot: 独热编码
            boolean，是否使用独热编码
        flat： 扁平化
            boolean，是否将图像数据扁平化成一维数据
    """

    def __init__(self, data_dir, mode='test', one_hot='false', flat='false'):
        super(Cifar10, self).__init__()
        self.p('data_dir', [{"path": data_dir}])
        self.p('mode', mode)
        self.p('one_hot', one_hot)
        self.p('flat', flat)

    def run(self):
        param = self.params

        from tfos.data.cifar import Cifar10
        from tfos.choices import DATA_MODE, BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        data_dir = param.get('data_dir')[0]['path']
        mode = param.get('mode', DATA_MODE[0])
        one_hot = param.get('one_hot', BOOLEAN[1])
        flat = param.get('flat', BOOLEAN[1])

        cifar = Cifar10(sc=sc, path=data_dir, one_hot=convert_bool(one_hot), flat=convert_bool(flat))
        output_df = cifar.train_df if mode == 'train' else cifar.test_df
        output_df.persist()
        output_df.show()
        outputRDD('<#zzjzRddName#>_cifar10_{}'.format(mode), output_df)


class Cifar100(Base):
    """cifar10数据集

    读取cifar-10数据集算子

    参数：
        data_dir: 数据目录
            cifar-100数据存放的目录
        mode: 读取模式
            读取数据的模式，分为训练数据和测试数据， train|test
        one_hot: 独热编码
            boolean，是否使用独热编码
        flat： 扁平化
            boolean，是否将图像数据扁平化成一维数据
    """

    def __init__(self, data_dir, mode='test', one_hot='false', flat='false'):
        super(Cifar100, self).__init__()
        self.p('data_dir', [{"path": data_dir}])
        self.p('mode', mode)
        self.p('one_hot', one_hot)
        self.p('flat', flat)

    def run(self):
        param = self.params

        from tfos.data.cifar import Cifar100
        from tfos.choices import DATA_MODE, BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        data_dir = param.get('data_dir')[0]['path']
        mode = param.get('mode', DATA_MODE[0])
        one_hot = param.get('one_hot', BOOLEAN[1])
        flat = param.get('flat', BOOLEAN[1])

        cifar = Cifar100(sc=sc, path=data_dir, one_hot=convert_bool(one_hot), flat=convert_bool(flat))
        output_df = cifar.train_df if mode == 'train' else cifar.test_df
        output_df.persist()
        output_df.show()
        outputRDD('<#zzjzRddName#>_cifar100_{}'.format(mode), output_df)


class TestCifar(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        path = ROOT_PATH if self.is_local else HDFS
        self.cifar10_dir = os.path.join(path, 'data/data/cifar10/cifar-10-batches-py')
        self.cifar100_dir = os.path.join(path, 'data/data/cifar100/cifar-100-python')

    # @unittest.skip('')
    def test_cifar10(self):
        data_mode = 'train'
        Cifar10(self.cifar10_dir, data_mode, 'true', 'true').run()
        Cifar10(self.cifar10_dir, data_mode, 'true', 'false').run()
        Cifar10(self.cifar10_dir, data_mode, 'false', 'false').run()
        Cifar10(self.cifar10_dir, data_mode, 'false', 'true').run()

    # @unittest.skip('')
    def test_cifar100(self):
        data_mode = 'train'
        Cifar100(self.cifar100_dir, data_mode, 'true', 'true').run()
        Cifar100(self.cifar100_dir, data_mode, 'true', 'false').run()
        Cifar100(self.cifar100_dir, data_mode, 'false', 'false').run()
        Cifar100(self.cifar100_dir, data_mode, 'false', 'true').run()


if __name__ == '__main__':
    unittest.main()
