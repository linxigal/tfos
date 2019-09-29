# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.keras.datasets.mnist import load_data

from tfos.data import BaseData


class Mnist(BaseData):
    def __init__(self, data_format='tfr', **kwargs):
        self.data_format = data_format
        super(Mnist, self).__init__(**kwargs)

        self.__mode = None
        self.height = 28
        self.width = 28
        self.channel = 1
        self.num_class = 10

    @property
    def train_df(self):
        self.__mode = 'train'
        rdd = self.load_rdd()
        return self.rdd2df(rdd)

    @property
    def test_df(self):
        self.__mode = 'test'
        rdd = self.load_rdd()
        return self.rdd2df(rdd)

    def load_rdd(self):
        if self.data_format == 'tfr':
            rdd = self.load_tfr()
        elif self.data_format == 'csv':
            rdd = self.load_csv()
        elif self.data_format == 'pickle':
            rdd = self.load_pickle()
        elif self.data_format == 'gz':
            rdd = self.load_gz()
        elif self.data_format == 'npz':
            rdd = self.load_npz()
        else:
            raise ValueError('unknown data format!')

        if self.flat:
            rdd = rdd.map(self.convert_flatten)
        else:
            rdd = rdd.map(self.convert_conv)

        if self.one_hot:
            rdd = rdd.map(self.convert_one)

        rdd = rdd.map(lambda x: (x[0] / 255.0, x[1]))
        rdd = rdd.map(self.to_string)
        return rdd

    @property
    def data_dir(self):
        return os.path.join(self.path, self.data_format)

    def load_tfr(self):
        data_dir = os.path.join(self.data_dir, self.__mode)
        tfr_rdd = self.sc.newAPIHadoopFile(data_dir, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                           keyClass="org.apache.hadoop.io.BytesWritable",
                                           valueClass="org.apache.hadoop.io.NullWritable")
        return tfr_rdd.map(lambda x: Mnist.tfr2sample(bytes(x[0])))

    def load_csv(self):
        image_path = os.path.join(self.data_dir, self.__mode, 'images')
        label_path = os.path.join(self.data_dir, self.__mode, 'labels')
        image_rdd = self.sc.textFile(image_path).map(self.from_csv)
        label_rdd = self.sc.textFile(label_path).map(self.from_csv)
        return image_rdd.zip(label_rdd)

    def load_pickle(self):
        image_path = os.path.join(self.data_dir, self.__mode, 'images')
        label_path = os.path.join(self.data_dir, self.__mode, 'labels')
        image_rdd = self.sc.pickleFile(image_path)
        label_rdd = self.sc.pickleFile(label_path)
        return image_rdd.zip(label_rdd)

    def load_gz(self):
        result = []
        data = {
            'train': ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'],
            'test': ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        }
        image_path = os.path.join(self.data_dir, data[self.__mode][0])
        label_path = os.path.join(self.data_dir, data[self.__mode][1])

        with open(image_path, 'rb') as f:
            result.append(extract_images(f))

        with open(label_path, 'rb') as f:
            result.append(extract_labels(f, self.one_hot))

        return self.sc.parallelize(zip(*result))

    def load_npz(self):
        path = os.path.join(self.data_dir, 'mnist.npz')
        train_data, test_data = load_data(path)

        if self.__mode == 'train':
            return self.sc.parallelize(zip(*train_data))
        else:
            return self.sc.parallelize(zip(*train_data))

    @staticmethod
    def convert_conv(row):
        x = np.reshape(row[0], (28, 28, 1))
        return x, row[1]

    @staticmethod
    def convert_flatten(row):
        x = np.reshape(row[0], (784,))
        return x, row[1]

    @staticmethod
    def convert_one(row):
        y = row[1]
        if not isinstance(row[1], np.ndarray):
            y = np.zeros([10])
            y[row[1]] = 1
        return row[0], y

    @staticmethod
    def to_string(row):
        x, y = row
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        return x, y

    @staticmethod
    def from_csv(s):
        return np.array([float(x) for x in s.split(',') if len(s) > 0])

    @staticmethod
    def tfr2sample(byte_str):
        example = tf.train.Example()
        example.ParseFromString(byte_str)
        features = example.features.feature
        image = np.array(features['image'].int64_list.value)
        label = np.array(features['label'].int64_list.value)
        return image, label
