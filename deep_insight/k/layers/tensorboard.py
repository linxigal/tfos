#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/8 10:47
:File   :tensorboard.py
:content:
  
"""

from deep_insight import *
from deep_insight.base import *


class TensorBoard(Base):
    """TensorBoard层

    参数：
        log_dir: 日志目录
            tensorboard日志目录，可以是本地目录，也可以是hdfs目录
        operator: 操作
            可选操作类型，可选start|stop

    """

    def __init__(self, log_dir, operator):
        super(TensorBoard, self).__init__()
        self.p('log_dir', [{"path": log_dir}])
        self.p('operator', operator)

    def run(self):
        param = self.params
        from tfos.k.layers import TensorBoardLayer
        from tfos.choices import OPERATORS

        # param = json.loads('<#zzjzParam#>')
        log_dir = param.get("log_dir")[0]['path']
        operator = param.get("operator", OPERATORS[0])

        if operator not in OPERATORS:
            raise ValueError("TensorBoard operator incorrect!!!")

        output_df = TensorBoardLayer(sqlc=sqlc, log_dir=log_dir).add(operator)
        if output_df:
            output_df.show()
            outputRDD('<#zzjzRddName#>_TensorBoard', output_df)


class TestTensorBoard(unittest.TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.log_dir = os.path.join(self.path, 'data/model/mnist_mlp/tensorboard')

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    # @unittest.skip('')
    def test_tensor_board_start(self):
        TensorBoard(self.log_dir, 'start').run()

    @unittest.skip('')
    def test_tensor_board_stop(self):
        TensorBoard(self.log_dir, 'stop').run()


if __name__ == '__main__':
    unittest.main()
