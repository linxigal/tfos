#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 15:43
:File   : load_mnist.py
"""
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec


class mnist():
    def __init__(self, path):
        self.X_dim = 784
        self.z_dim = 100
        self.y_dim = 10
        self.size = 28
        self.channel = 1
        self.data = input_data.read_data_sets(path, one_hot=True)

    def __call__(self, batch_size):
        images, labels = self.data.train.next_batch(batch_size)
        return images, labels

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size, self.size), cmap='Greys_r')
        return fig
