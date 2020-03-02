#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/27 15:24
:File   :cifar10.py
:content:

"""

from deep_insight.base import *


class CrfWord2id(Base):
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

    def __init__(self, data_dir=None):
        super(CrfWord2id, self).__init__()
        # self.p('data_dir', [{"path": data_dir}])
        # self.p('mode', mode)
        self.p('data_dir', data_dir)

    def run(self):
        param = self.params

        from tfos.data.crf_word2id import CrfWord2id

        # param = json.loads('<#zzjzParam#>')
        # data_dir = param.get('data_dir')[0]['path']
        data_dir = param.get('data_dir')

        crf_word2id = CrfWord2id(sc=sc, path=data_dir)
        output_df = crf_word2id.train_df
        output_df.persist()
        output_df.show()
        outputRDD('<#zzjzRddName#>_crf_word2id', output_df)


class TestCrfWord2id(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        # path = ROOT_PATH if self.is_local else HDFS
        # self.cifar100_dir = os.path.join(path, 'data/data/cifar100/cifar-100-python')
        # local data
        self.data_dir = "data/data/text/test_data.data"

    # @unittest.skip('')
    def test_crf_word2id(self):
        CrfWord2id(data_dir=self.data_dir).run()
        # data_mode = 'train'
        # Cifar10(self.cifar10_dir, data_mode, 'true', 'true').run()


if __name__ == '__main__':
    unittest.main()
