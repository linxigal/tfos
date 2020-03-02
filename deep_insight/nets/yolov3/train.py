#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/25 16:39
:File   :train.py
:content:
  
"""

from deep_insight import *
from deep_insight.base import *
from deep_insight.nets.yolov3.yolov3 import YOLOV3Model, YOLOV3TinyModel


class YOLOV3Train(Base):

    def __init__(self, cluster_size, num_ps, data_dir, batch_size, epochs,
                 image_size, model_dir, weights_path=None, freeze_body=2, go_on='false', **kwargs):
        super(YOLOV3Train, self).__init__()
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('data_dir', data_dir)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        self.p('image_size', image_size)
        self.p('model_dir', model_dir)
        self.p('freeze_body', freeze_body)
        self.p('go_on', go_on)

    def run(self):
        param = self.params

        from tfos.nets.yolov3.tfos import YoloTFOS
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get('input_prev_layers')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        data_dir = param.get('data_dir')
        batch_size = param.get('batch_size', 32)
        epochs = param.get('epochs', 1)
        image_size = param.get('image_size')
        model_dir = param.get('model_dir')
        freeze_body = param.get('freeze_body', '2')
        go_on = param.get('go_on', BOOLEAN[1])

        model_rdd = inputRDD(input_prev_layers)
        print(model_rdd)
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)

        kwargs = dict()
        kwargs['model_rdd'] = model_rdd
        kwargs['data_dir'] = data_dir
        kwargs['batch_size'] = int(batch_size)
        kwargs['epochs'] = int(epochs)
        kwargs['image_size'] = tuple(int(x) for x in image_size.split(','))
        kwargs['model_dir'] = model_dir
        if freeze_body:
            kwargs['freeze_body'] = int(freeze_body)
        kwargs['go_on'] = convert_bool(go_on)

        output_df = YoloTFOS(sc, sqlc, cluster_size, num_ps).yolov3_train(**kwargs)
        # output_df.show()
        outputRDD('<#zzjzRddName#>_yolov3', output_df)


class YOLOV3TinyTrain(Base):

    def __init__(self, cluster_size, num_ps, batch_size, epochs, classes_path, anchors_path, train_path, val_path,
                 image_size, model_dir, weights_path=None, freeze_body=2, go_on='false', **kwargs):
        super(YOLOV3TinyTrain, self).__init__()
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        self.p('classes_path', classes_path)
        self.p('anchors_path', anchors_path)
        self.p('weights_path', weights_path)
        self.p('train_path', train_path)
        self.p('val_path', val_path)
        self.p('image_size', image_size)
        self.p('model_dir', model_dir)
        self.p('freeze_body', freeze_body)
        self.p('go_on', go_on)

    def run(self):
        param = self.params

        from tfos.nets.yolov3.tfos import YoloTFOS
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get('input_prev_layers')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        batch_size = param.get('batch_size', 32)
        epochs = param.get('epochs', 1)
        image_size = param.get('image_size')
        model_dir = param.get('model_dir')
        freeze_body = param.get('freeze_body', '2')
        go_on = param.get('go_on', BOOLEAN[1])

        model_rdd = inputRDD(input_prev_layers)
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)

        kwargs = dict()
        kwargs['model_rdd'] = model_rdd
        kwargs['batch_size'] = int(batch_size)
        kwargs['epochs'] = int(epochs)
        kwargs['model_dir'] = model_dir
        if freeze_body:
            kwargs['freeze_body'] = int(freeze_body)
        kwargs['go_on'] = convert_bool(go_on)

        output_df = YoloTFOS(sc, sqlc, cluster_size, num_ps).yolov3_tiny_train(**kwargs)
        output_df.show()
        outputRDD('<#zzjzRddName#>_yolov3_tiny', output_df)


class TestYOLOV3Train(TestCase):

    def setUp(self):
        self.is_local = True
        self.data_dir = self.join('data/data/VOCdevkit/model_data')
        self.model_dir = self.join('data/model/yolov3')

    def tearDown(self):
        reset()

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    # @unittest.skip('')
    def test_yolov3_train(self):
        YOLOV3Model('9', '20').run()
        YOLOV3Train(cluster_size='3',
                    num_ps='1',
                    data_dir=self.data_dir,
                    batch_size='4',
                    epochs='1',
                    image_size='416,416',
                    go_on='false',
                    model_dir=self.model_dir).run()

    @unittest.skip('')
    def test_yoloV3_tiny_train(self):
        YOLOV3TinyModel('6', '20').run()
        YOLOV3TinyTrain(cluster_size='3',
                        num_ps='1',
                        batch_size='4',
                        epochs='1',
                        image_size='416,416',
                        model_dir=self.model_dir).run()


if __name__ == '__main__':
    unittest.main()
