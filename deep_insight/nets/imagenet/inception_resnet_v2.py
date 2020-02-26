#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :huangdehua
:Time:  :2019/12/06 17:05
:File   :inception_resnet_v2.py
:content:

"""


from deep_insight.base import *
from deep_insight.data.cifar import Cifar10
from deep_insight.compile import CompileAdvanced,Compile
from deep_insight.model.model import TrainModel
from tfos.base import logger


class InceptionResnetV2(Base):
    """InceptionResnetV2模型
        InceptionResnetV2模型是google推出的深度CNN模型，结合了ResNet与GoogleNet。

        InceptionResnetV2是一个完整的模型

        参数
            输入维度：input_shape
                大或等于0的整数，字典长度，即输入数据最大下标+1
            维度转换: reshape
                嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
            输出维度：out_dense
                大于0的整数，代表全连接嵌入的维度


        输入shape
            形如（samples，sequence_length）的2D张量

        输出shape
            形如(samples, sequence_length, output_dim)的3D张量
        """

    def __init__(self, input_shape='32,32,3', reshape=None,
                 out_dense='10'):
        super(InceptionResnetV2, self).__init__()
        self.p('input_shape', input_shape)
        self.p('reshape', reshape)
        self.p('out_dense', out_dense)

    def run(self):
        param = self.params
        from tfos.layers import EmbeddingLayer
        from tfos.utils import convert_bool
        from tfos.choices import BOOLEAN
        from tfos.nets.imagenet.inception_resnet_v2 import InceptionResnetV2

        # param = json.loads('<#zzjzParam#>')
        input_shape = param.get("input_shape")
        reshape = param.get("reshape")
        out_dense = param.get("out_dense")

        input_shape = tuple(input_shape.split(','))
        #reshape = tuple(reshape.split(','))
        out_dense = int(out_dense)
        output_df = InceptionResnetV2(sqlc=sqlc).add(input_shape, reshape, out_dense)
        outputRDD('<#zzjzRddName#>_InceptionResnetV2_mode', output_df)
        output_df.show()

class TestInput(TestCase):

    def setUp(self) -> None:
        self.is_local = False
        self.data_dir = os.path.join(self.path, 'data/data/cifar10')
        self.model_dir = os.path.join(self.path, 'data/model/inception_resnet_v2_cifar10')

    # @unittest.skip("")
    def test_input(self):
        Cifar10(self.data_dir, one_hot=True, mode='test').b(DATA_BRANCH).run()
        InceptionResnetV2().b(MODEL_BRANCH).run()
        #Compile().run()
        Compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']).run()
        TrainModel(
            input_prev_layers=MODEL_BRANCH,
            input_rdd_name=DATA_BRANCH,
            cluster_size=3,
            num_ps=1,
            batch_size=64,
            epochs=2,
            #model_dir='hdfs://master:8020/data/model/inception_resnet_v2_cifar10',
            model_dir=self.model_dir,
            go_on='false').run()


if __name__ == '__main__':
    unittest.main()
