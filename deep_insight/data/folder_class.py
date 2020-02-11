# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

from deep_insight.base import *


class FolderClass(Base):
    """文件夹归类图片数据集

    参数：
        data_dir： 数据集目录
        data_mode: 读取模式
            读取数据的模式，分为训练数据和测试数据， train|test
        split_ratio： 数据集切分比例
            按照此参数设置的比例，将数据集切分成训练集和测试集
        min_num_per_class： 每个类别最小数据量
            每一个类别下面的图片数量不能小于一张图片
        mode： 数据集切分模式
            有两种模式，'SPLIT_IMAGES'按照图片总的数量切分，'SPLIT_CLASSES'按照图片的类别数量切分

    """

    def __init__(self, data_dir, data_mode='train', split_ratio='0.0', min_num_per_class='1', mode='SPLIT_IMAGES'):
        super(FolderClass, self).__init__()
        self.p('data_dir', data_dir)
        self.p('data_mode', data_mode)
        self.p('split_ratio', split_ratio)
        self.p('min_num_per_class', min_num_per_class)
        self.p('mode', mode)

    def run(self):
        param = self.params

        from tfos.data.folder_class import FolderClass
        from tfos.choices import SPLIT_MODE, DATA_MODE

        # param = json.loads('<#zzjzParam#>')
        data_dir = param.get('data_dir')
        data_mode = param.get('data_mode', DATA_MODE[0])
        split_ratio = param.get('split_ratio', 0.0)
        min_num_per_class = param.get('min_num_per_class', 1)
        mode = param.get('mode', SPLIT_MODE[0])

        kwargs = dict(data_dir=data_dir)
        if split_ratio:
            split_ratio = float(split_ratio)
            assert 0 <= split_ratio <= 1
            kwargs['split_ratio'] = split_ratio
        if min_num_per_class:
            min_num_per_class = int(min_num_per_class)
            assert min_num_per_class > 0
            kwargs['min_num_per_class'] = min_num_per_class
        if mode:
            kwargs['mode'] = mode

        if data_mode == 'train':
            n_classes, data_rdd, mark_rdd = FolderClass(sc, **kwargs).train_data
        else:
            n_classes, data_rdd, mark_rdd = FolderClass(sc, **kwargs).test_data
        print(n_classes)
        data_rdd.show()
        mark_rdd.show()
        output_name = '<#zzjzRddName#>_{}_'.format(data_mode)
        outputRDD(output_name + 'n_classes', n_classes)
        outputRDD(output_name + 'data', data_rdd)
        outputRDD(output_name + 'mark', mark_rdd)


class TestFolderClass(TestCase):

    def setUp(self) -> None:
        self.is_local = True
        self.data_dir = os.path.join(self.path, "data/data/lfw/lfw_160_30/images")

    # @unittest.skip('')
    def test_lfw(self):
        FolderClass(self.data_dir).run()
        FolderClass(self.data_dir, split_ratio='0.8').run()
        FolderClass(self.data_dir, split_ratio='0.8', mode='SPLIT_CLASSES').run()


if __name__ == '__main__':
    unittest.main()
