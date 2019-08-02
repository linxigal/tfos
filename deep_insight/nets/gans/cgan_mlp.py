#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/7/22 11:18
:File   : cgan_mlp.py
:content:
    standalone schema execute:
        spark-submit    --master ${MASTER} \
                        deep_insight/nets/gans/cgan_mlp.py \
                        --data_dir /home/wjl/github/tfos/data/mnist \
                        --output_dir /home/wjl/github/tfos/data/results \
                        --ckpt_dir /home/wjl/github/tfos/data/checkpoint \
                        --steps 10000 \
                        --batch_size 32
    yarn schema execute:
        spark-submit    --master yarn \
                        deep_insight/nets/gans/cgan_mlp.py \
                        --data_dir hdfs://t-master:8020/data/model/mnist_cgan_mlp/mnist \
                        --output_dir hdfs://t-master:8020/data/model/mnist_cgan_mlp/results \
                        --ckpt_dir hdfs://t-master:8020/data/model/mnist_cgan_mlp/checkpoint \
                        --steps 10000 \
                        --batch_size 32
"""

import argparse
import os

from deep_insight.base import *
from tfos import ROOT_PATH


class CGAN_MLP(Base):
    def __init__(self, data_path, output_path, cluster_size, num_ps, steps, batch_size):
        super(CGAN_MLP, self).__init__()
        self.p('data_path', [{"path": data_path}])
        self.p('output_path', [{"path": output_path}])
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('steps', steps)
        self.p('batch_size', batch_size)

    def run(self):
        param = self.params

        from tfos.nets.gans.cgan_mlp import TFOS_CGAN_MLP
        from tfos.data.load_mnist import mnist

        # param = json.loads('<#zzjzParam#>')
        data_path = param.get('data_path')[0]['path']
        output_path = param.get('output_path')[0]['path']
        cluster_size = int(param.get('cluster_size'))
        num_ps = int(param.get('num_ps'))
        steps = int(param.get('steps'))
        batch_size = int(param.get('batch_size'))
        # TFOS_CGAN_MLP(sc, cluster_size, num_ps).train(mnist(data_path), output_path, steps, batch_size)
        TFOS_CGAN_MLP.local_train(mnist(data_path), output_path, steps, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")

    parser.add_argument("--data_path", help="HDFS path to train data")
    parser.add_argument("--output_path", help="HDFS path to train model output result")
    parser.add_argument("--cluster_size", help="number of cluster size", type=int, default=3)
    parser.add_argument("--num_ps", help="number of num of parameter server", type=int, default=1)
    parser.add_argument("--steps", help="number of epochs", type=int, default=10000)
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=32)
    args = parser.parse_args()
    CGAN_MLP(**vars(args)).run()
