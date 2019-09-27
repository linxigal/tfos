#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/20 15:31
:File   :mnist.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class Mnist(Base):
    """Mnist数据集

    参数：
        mnist_dir: 数据目录
            mnist数据集的目录，目录中的子目录以数据类型命名，目前支持五种格式
        data_format: 数据格式
            数据格式，可选项包括tfr|csv|pickle|gz|npz
        one_hot: 独热编码
        is_conv: 卷积数据
            是否是卷积类型：shape=(28,28,1)，否则: shape=(1,784)
        mode: 数据集
            包括两种类型的数据集，train和test
    """

    def __init__(self, mnist_dir, data_format='tfr', one_hot='true', is_conv='False', mode='test'):
        super(Mnist, self).__init__()
        self.p('mnist_dir', [{"path": mnist_dir}])
        self.p('data_format', data_format)
        self.p('one_hot', one_hot)
        self.p('is_conv', is_conv)
        self.p('mode', mode)

    def run(self):
        param = self.params

        from tfos.data.mnist import Mnist
        from tfos.utils import convert_bool
        from tfos.choices import BOOLEAN, MNIST_FORMAT, DATA_MODE

        # param = json.loads('<#zzjzParam#>')
        mnist_dir = param.get('mnist_dir')[0]['path']
        mode = param.get('mode', DATA_MODE[0])
        data_format = param.get('data_format', MNIST_FORMAT[0])
        one_hot = param.get('one_hot', BOOLEAN[0])
        is_conv = param.get('is_conv', BOOLEAN[1])

        one_hot = convert_bool(one_hot)
        is_conv = convert_bool(is_conv)

        mnist = Mnist(sc, mnist_dir, data_format, one_hot, is_conv)
        output_df = mnist.train_df if mode == 'train' else mnist.test_df
        outputRDD('<#zzjzRddName#>_mnist_{}'.format(mode), output_df)


class TestMnist(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        path = ROOT_PATH if self.is_local else HDFS
        self.mnist_dir = os.path.join(path, 'data/data/mnist')

    @unittest.skip('')
    def test_mnist_tfr(self):
        Mnist(self.mnist_dir, 'tfr', 'true', 'true').run()
        Mnist(self.mnist_dir, 'tfr', 'false', 'false').run()
        Mnist(self.mnist_dir, 'tfr', 'true', 'false').run()
        Mnist(self.mnist_dir, 'tfr', 'false', 'true').run()

    @unittest.skip('')
    def test_mnist_csv(self):
        Mnist(self.mnist_dir, 'csv', 'true', 'true').run()
        Mnist(self.mnist_dir, 'csv', 'false', 'false').run()
        Mnist(self.mnist_dir, 'csv', 'true', 'false').run()
        Mnist(self.mnist_dir, 'csv', 'false', 'true').run()

    @unittest.skip('')
    def test_mnist_pickle(self):
        Mnist(self.mnist_dir, 'pickle', 'true', 'true').run()
        Mnist(self.mnist_dir, 'pickle', 'false', 'false').run()
        Mnist(self.mnist_dir, 'pickle', 'true', 'false').run()
        Mnist(self.mnist_dir, 'pickle', 'false', 'true').run()

    @unittest.skip('')
    def test_mnist_gz(self):
        Mnist(self.mnist_dir, 'gz', 'true', 'true').run()
        Mnist(self.mnist_dir, 'gz', 'false', 'false').run()
        Mnist(self.mnist_dir, 'gz', 'true', 'false').run()
        Mnist(self.mnist_dir, 'gz', 'false', 'true').run()

    @unittest.skip('')
    def test_mnist_npz(self):
        Mnist(self.mnist_dir, 'npz', 'true', 'true').run()
        Mnist(self.mnist_dir, 'npz', 'false', 'false').run()
        Mnist(self.mnist_dir, 'npz', 'true', 'false').run()
        Mnist(self.mnist_dir, 'npz', 'false', 'true').run()


if __name__ == '__main__':
    unittest.main()
