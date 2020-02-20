# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

from deep_insight.base import *


class MLPModel(Base):
    """多层感知机模型层

    参数：
        keep_prob：节点失活率

    """

    def __init__(self, keep_prob='0.8'):
        super(MLPModel, self).__init__()
        self.p('keep_prob', keep_prob)

    def run(self):
        param = self.params

        from tfos.tf.models.mlp import MLPModel
        from tfos.tf import TFModeMiddle

        # param = json.loads('<#zzjzParam#>')
        keep_prob = param.get('keep_prob')

        keep_prob = float(keep_prob)
        assert 0 <= keep_prob <= 1, ""

        output_df = TFModeMiddle(MLPModel(keep_prob), sqlc).serialize
        output_df.show()
        outputRDD('<#zzjzRddName#>_mlp_model', output_df)


class MLPCompile(Base):
    """多层感知机编译层

    """

    def __init__(self):
        super(MLPCompile, self).__init__()

    def run(self):
        param = self.params

        from tfos.tf.models.mlp import MLPCompile
        from tfos.tf import TFModeMiddle

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get('input_prev_layers')

        model_rdd = inputRDD(input_prev_layers)
        assert model_rdd, "cannot get model config rdd from previous model layer!"

        output_df = TFModeMiddle(MLPCompile(), sqlc, model_rdd).serialize
        output_df.show()
        outputRDD('<#zzjzRddName#>_mlp_compile', output_df)


class TestMLP(TestCase):

    def test_mlp(self):
        MLPModel().run()
        MLPCompile().run()


if __name__ == '__main__':
    unittest.main()
