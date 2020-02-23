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
        input_dim： 输入维度
        hidden_units：隐藏单元
        keep_prob：节点存活率

    """

    def __init__(self, keep_prob='0.8'):
        super(MLPModel, self).__init__()
        self.p('keep_prob', keep_prob)

    def run(self):
        param = self.params

        from tfos.tf.models.mlp import MLPModel
        from tfos.tf import TFModeMiddle

        # param = json.loads('<#zzjzParam#>')
        input_dim = param.get('input_dim', 784)
        hidden_units = param.get('hidden_units', 300)
        keep_prob = param.get('keep_prob', 0.8)

        kwargs = {}
        if input_dim:
            input_dim = int(input_dim)
            assert input_dim > 0
            kwargs['input_dim'] = input_dim
        if hidden_units:
            hidden_units = int(hidden_units)
            assert hidden_units > 0
            kwargs['hidden_units'] = hidden_units
        if keep_prob:
            keep_prob = float(keep_prob)
            assert 0 <= keep_prob <= 1
            kwargs['keep_prob'] = keep_prob
        output_df = TFModeMiddle(MLPModel(**kwargs), sqlc).serialize
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
