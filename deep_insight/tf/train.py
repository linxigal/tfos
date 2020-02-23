# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

from deep_insight.base import *
from deep_insight.data.mnist import Mnist
from deep_insight.tf.models.mlp import MLPModel, MLPCompile


class TFTrain(Base):
    """TF模型训练
    基于TensorFlow的模型的训练算子

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
        super(TFTrain, self).__init__(**kwargs)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        self.p('model_dir', [{"path": model_dir}])
        self.p('go_on', go_on)

    def run(self):
        param = self.params

        from tfos.tf.tfos import TFOnSpark
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
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        # load model
        model_rdd = inputRDD(input_prev_layers)
        assert model_rdd, "cannot get model config rdd from previous model layer!"

        args = input_rdd, model_rdd, batch_size, epochs, model_dir, go_on
        output_df = TFOnSpark(sc, sqlc, cluster_size, num_ps).train(*args)
        output_df.show()
        outputRDD('<#zzjzRddName#>_train_result', output_df)


class TFPredict(Base):
    """TF模型预测

    基于TensorFlow的模型的预测算子

    参数：
        input_rdd_name: 输入数据
            输入的RDD格式数据
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务器ps和计算服务器worker
        num_ps: 参数服务器数量
        steps: 预测数据条数
            预测多少条数据
        params：模型参数
            模型预测时需要输入的额外参数，可以查看模型的配置文件
        model_dir: 模型保存路径
            保存路径下会自动生成tensorboard目录，checkpoint目录以及save_model目录

    """

    def __init__(self, cluster_size, num_ps, steps, params, model_dir):
        super(TFPredict, self).__init__()
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('steps', steps)
        self.p('params', params)
        self.p('model_dir', [{"path": model_dir}])

    def run(self):
        param = self.params

        import json
        from tfos.tf.tfos import TFOnSpark

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        steps = param.get('steps', 0)
        params = param.get('params', '')
        model_dir = param.get('model_dir')[0]['path']

        cluster_size = int(cluster_size)
        num_ps = int(num_ps)
        steps = int(steps)
        params = params.strip()

        if params:
            params = json.loads(params)

        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"

        args = input_rdd, steps, model_dir, params
        output_df = TFOnSpark(sc, sqlc, cluster_size, num_ps).predict(*args)
        output_df.show()
        outputRDD('<#zzjzRddName#>_predict_result', output_df)


class TestTFTrain(TestCase):

    def setUp(self) -> None:
        self.is_local = True
        self.mnist_dir = os.path.join(self.path, 'data/data/mnist')
        self.model_dir = os.path.join(self.path, 'data/model/tf_mnist_mlp')

    @unittest.skip('')
    def test_tf_train(self):
        Mnist(self.mnist_dir, mode='train').b(DATA_BRANCH).run()
        MLPModel().b(MODEL_BRANCH).run()
        MLPCompile().run()
        TFTrain(input_prev_layers=MODEL_BRANCH,
                input_rdd_name=DATA_BRANCH,
                cluster_size=3,
                num_ps=1,
                batch_size=32,
                epochs=2,
                model_dir=self.model_dir,
                go_on='false').run()

    # @unittest.skip('')
    def test_tf_predict(self):
        Mnist(self.mnist_dir, mode='test').b(DATA_BRANCH).run()
        TFPredict(cluster_size=3,
                  num_ps=1,
                  steps=10,
                  params='{"keep_prob": 1}',
                  model_dir=self.model_dir).run()


if __name__ == '__main__':
    unittest.main()
