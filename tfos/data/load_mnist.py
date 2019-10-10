#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 15:43
:File   : mnist.py
"""

import os
import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.datasets.mnist import load_data


# class Mnist(object):
#     def __init__(self, sc, mnist_dir, format='gz', one_hot=True, is_conv=False):
#         self.sc = sc
#         self.mnist_dir = mnist_dir
#         self.format = format
#         self.one_hot = one_hot
#         self.is_conv = is_conv
#
#     def load_data(self):
#         train_data, test_data = [], []
#         if self.format == 'gz':
#             train_data, test_data = self.load_gz()
#         elif self.format == 'npz':
#             train_data, test_data = self.load_npz()
#
#         train_rdd = self.sc.parallelize(zip(*self.valid(train_data)))
#         test_rdd = self.sc.parallelize(zip(*self.valid(test_data)))
#         return train_rdd, test_rdd
#
#     def valid(self, features, labels):
#         if not self.is_conv:
#             features = features.reshape(None, 784)
#         return features, labels
#
#     def load_npz(self):
#         (x_train, y_train), (x_test, y_test) = load_data(os.path.join(self.mnist_dir, 'npz/mnist.npz'))
#         if not self.is_conv:
#             x_train = x_train.reshape(None, 784)
#             x_test = x_train.reshape(None, 784)
#
#     def load_gz(self):
#         train_data = []
#         test_data = []
#         with open(os.path.join(self.mnist_dir, 'gz', 'train-images-idx3-ubyte.gz'), 'rb') as f:
#             train_data.append(mnist.extract_images(f))
#         with open(os.path.join(self.mnist_dir, 'gz', 'train-labels-idx1-ubyte.gz'), 'rb') as f:
#             train_data.append(mnist.extract_labels(f, self.one_hot))
#
#         with open(os.path.join(self.mnist_dir, 'gz', 't10k-images-idx3-ubyte.gz'), 'rb') as f:
#             test_data.append(mnist.extract_images(f))
#         with open(os.path.join(self.mnist_dir, 'gz', 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
#             test_data.append(mnist.extract_labels(f, self.one_hot))
#         return train_data, test_data


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
