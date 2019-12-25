#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/3 10:35
:File   :model.py
:content: neural network model train,inference and predict
    local test execute:
         spark-submit  \
         --jars /home/wjl/github/TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
         deep_insight/model/model.py
"""

import unittest

from deep_insight import *
from deep_insight.base import *
from deep_insight.compile import Compile
from deep_insight.data.mnist import Mnist
from deep_insight.layers.core import Dropout, Dense
from deep_insight.layers.input import InputLayer


class TrainModel(Base):
    """模型训练
    神经网络模型训练算子

    参数：
        input_rdd_name: 输入数据
            输入的RDD格式数据
        input_config: 输入模型
            输入的RDD模型配置数据，包括模型的图结构以及模型的编译优化参数
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务器ps和计算服务器worker
        num_ps: 参数服务器数量
        batch_size： 批处理大小
        epochs：数据集迭代次数
            对传入的数据集，训练模型时需要迭代的次数
        model_dir: 模型保存路径
            保存路径下会自动生成tensorboard目录，checkpoint目录以及save_model目录
        go_on: 继续训练
            boolean， 是否接着上次训练结果继续训练模型
    """

    def __init__(self, cluster_size, num_ps, batch_size, epochs, model_dir, go_on='false', **kwargs):
        super(TrainModel, self).__init__(**kwargs)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        # self.p('steps_per_epoch', steps_per_epoch)
        self.p('model_dir', [{"path": model_dir}])
        self.p('go_on', go_on)

    def run(self):
        param = self.params

        from tfos import TFOS
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        input_prev_layers = param.get('input_prev_layers')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        batch_size = param.get('batch_size', 32)
        epochs = param.get('epochs', 1)
        model_dir = param.get('model_dir')[0]['path']
        go_on = param.get('go_on', BOOLEAN[1])

        # param check
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)
        batch_size = int(batch_size)
        epochs = int(epochs)
        go_on = convert_bool(go_on)

        # load data
        assert input_rdd_name is not None, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        # load model
        assert input_prev_layers is not None, "parameter input_model_config cannot empty!"
        model_rdd = inputRDD(input_prev_layers)
        assert model_rdd, "cannot get model config rdd from previous model layer!"
        output_df = TFOS(sc, sqlc, cluster_size, num_ps).train(data_rdd=input_rdd,
                                                               model_rdd=model_rdd,
                                                               batch_size=batch_size,
                                                               epochs=epochs,
                                                               model_dir=model_dir,
                                                               go_on=go_on)
        if output_df:
            output_df.show()
            outputRDD('<#zzjzRddName#>_train_result', output_df)


class EvaluateModel(Base):
    """模型训练
    神经网络模型训练算子

    参数：
        input_rdd_name: 输入数据
            输入的RDD格式数据
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务器ps和计算服务器worker
        num_ps: 参数服务器数量
        steps: 执行步数
        model_dir: 模型保存路径
            保存路径下会自动生成tensorboard目录，checkpoint目录以及save_model目录
    """

    def __init__(self, cluster_size, num_ps, steps, model_dir, **kwargs):
        super(EvaluateModel, self).__init__(**kwargs)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('steps', steps)
        self.p('model_dir', [{"path": model_dir}])

    def run(self):
        param = self.params

        from tfos import TFOS

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        steps = param.get('steps', 0)
        model_dir = param.get('model_dir')[0]['path']

        # param check
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)
        steps = int(steps)

        # load data
        assert input_rdd_name is not None, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"

        output_df = TFOS(sc, sqlc, cluster_size, num_ps).evaluate(input_rdd, steps, model_dir)
        if output_df:
            output_df.show()
            outputRDD('<#zzjzRddName#>_evaluate_result', output_df)


class PredictModel(Base):
    """模型校验
    神经网络模型校验算子

    参数：
        input_rdd_name: 输入数据
            输入rdd校验数据
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务器ps和计算服务器worker
        num_ps: 参数服务器数量
        steps: 执行步数
        model_dir: 模型保存路径
            保存路径下会自动生成tensorboard目录，checkpoint目录以及save_model目录
        output_prob：输出预测值
            boolean，是否输出每个样本在所有分类类别上的详细概率值
    """

    def __init__(self, cluster_size, num_ps, steps, model_dir, output_prob='false', **kwargs):
        super(PredictModel, self).__init__(**kwargs)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('steps', steps)
        self.p('model_dir', [{"path": model_dir}])
        self.p('output_prob', output_prob)

    def run(self):
        param = self.params

        from tfos import TFOS
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        steps = param.get('steps', 0)
        model_dir = param.get('model_dir')[0]['path']
        output_prob = param.get('output_prob', BOOLEAN[0])

        # param check
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)
        steps = int(steps)
        output_prob = convert_bool(output_prob)

        # load data
        assert input_rdd_name is not None, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"

        output_df = TFOS(sc, sqlc, cluster_size, num_ps).predict(input_rdd, steps, model_dir, output_prob)
        if output_df:
            output_df.show()
            outputRDD('<#zzjzRddName#>_predict_result', output_df)


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = False
        self.mnist_dir = os.path.join(self.path, 'data/data/mnist')
        self.model_dir = os.path.join(self.path, 'data/model/mnist_mlp')

    def tearDown(self) -> None:
        reset()

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    @staticmethod
    def build_model():
        m = MODEL_BRANCH
        # build model
        InputLayer('784').b(m).run()
        Dense('512', activation='relu').run()
        # Dense('512', activation='relu', input_shape='784').b(m).run()
        Dropout('0.2').run()
        Dense('512', activation='relu').run()
        Dropout('0.2').run()
        Dense('10', activation='softmax').run()
        # compile model
        Compile('categorical_crossentropy', 'rmsprop', ['accuracy']).run()
        # show network struct
        SummaryLayer(m).run()

    # @unittest.skip("")
    def test_train_model(self):
        Mnist(self.mnist_dir, mode='train').b(DATA_BRANCH).run()
        self.build_model()
        TrainModel(input_prev_layers=MODEL_BRANCH,
                   input_rdd_name=DATA_BRANCH,
                   cluster_size=3,
                   num_ps=1,
                   batch_size=32,
                   epochs=2,
                   model_dir=self.model_dir,
                   go_on='false').run()

    @unittest.skip("")
    def test_evaluate_model(self):
        Mnist(self.mnist_dir, mode='test').b(DATA_BRANCH).run()
        EvaluateModel(input_prev_layers=MODEL_BRANCH,
                      input_rdd_name=DATA_BRANCH,
                      cluster_size=3,
                      num_ps=1,
                      steps=0,
                      model_dir=self.model_dir).run()

    @unittest.skip("")
    def test_predict_model(self):
        Mnist(self.mnist_dir, mode='test').b(DATA_BRANCH).run()
        PredictModel(input_prev_layers=MODEL_BRANCH,
                     input_rdd_name=DATA_BRANCH,
                     cluster_size=3,
                     num_ps=1,
                     steps=10,
                     model_dir=self.model_dir,
                     output_prob='true').run()


if __name__ == '__main__':
    unittest.main()
