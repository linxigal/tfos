#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/7/22 11:18
:File   : cgan_mlp.py
"""

import os
import shutil

from deep_insight.base import *
from tfos import ROOT_PATH


class CGAN_MLP(Base):
    def __init__(self, data_dir, output_dir, ckpt_dir, steps, batch_size):
        super(CGAN_MLP, self).__init__()
        self.p('data_dir', [{"path": data_dir}])
        self.p('output_dir', [{"path": output_dir}])
        self.p('ckpt_dir', [{"path": ckpt_dir}])
        self.p('steps', steps)
        self.p('batch_size', batch_size)

    def run(self):
        param = self.params

        from tfos.nets.gans.cgan_mlp import CGAN_MLP
        from tfos.data.load_mnist import mnist

        # param = json.loads('<#zzjzParam#>')
        data_dir = param.get('data_dir')[0]['path']
        output_dir = param.get('output_dir')[0]['path']
        ckpt_dir = param.get('ckpt_dir')[0]['path']
        steps = int(param.get('steps'))
        batch_size = int(param.get('batch_size'))
        CGAN_MLP(mnist(data_dir), output_dir, ckpt_dir).train(steps, batch_size)


if __name__ == "__main__":
    data_dir = os.path.join(ROOT_PATH, 'data', 'Mnist')
    output_dir = os.path.join(ROOT_PATH, 'data', 'results')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    ckpt_dir = os.path.join(ROOT_PATH, 'data', 'checkpoint')
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir)
    CGAN_MLP(data_dir, output_dir, ckpt_dir, 1000000, 32).run()
