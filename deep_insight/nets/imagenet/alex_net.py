from deep_insight.base import *
from deep_insight.data.cifar import Cifar10
from deep_insight.k.compile import Compile
from deep_insight.k.train import TrainModel


class AlexNet(Base):

    def __init__(self, input_shape='32,32,3', reshape=None,
                 out_dense='10'):
        super(AlexNet, self).__init__()
        self.p('input_shape', input_shape)
        self.p('reshape', reshape)
        self.p('out_dense', out_dense)

    def run(self):
        param = self.params

        from tfos.nets.imagenet.alex_net import AlexNet
        # param = json.loads('<#zzjzParam#>')
        input_shape = param.get('input_shape')
        reshape = param.get("reshape")
        out_dense = param.get('out_dense')

        input_shape = tuple([int(i) for i in input_shape.split(',') if i])
        # reshape = tuple(reshape.split(','))
        out_dense = int(out_dense)
        output_df = AlexNet(sqlc=sqlc).add(input_shape, reshape, out_dense)
        outputRDD('<#zzjzRddName#>_AlexNet_mode', output_df)
        output_df.show()


class TestInput(TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.data_dir = os.path.join(self.path, 'data/data/cifar10')
        self.model_dir = os.path.join(self.path, 'data/model/alex_net_cifar10')

    @unittest.skip("")
    def test_alex_net_v2(self):
        AlexNet().run()
        SummaryLayer().run()

    # @unittest.skip("")
    def test_alex_net_v2_train(self):
        Cifar10(self.data_dir, one_hot=True, mode='test').b(DATA_BRANCH).run()
        AlexNet().b(MODEL_BRANCH).run()
        Compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']).run()
        TrainModel(
            input_prev_layers=MODEL_BRANCH,
            input_rdd_name=DATA_BRANCH,
            cluster_size=2,
            num_ps=1,
            batch_size=32,
            epochs=5,
            model_dir=self.model_dir,
            go_on='false').run()


if __name__ == '__main__':
    unittest.main()
