# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import tensorflow as tf

from tfos.tf import TFMode, TFCompile
from tfos.base import logger


class MLPModel(TFMode):

    def __init__(self, keep_prob):
        super(MLPModel, self).__init__()
        self.add_params(keep_prob=keep_prob)

    def build_model(self):
        in_units = 784  # 输入节点数
        h1_units = 300  # 隐含层节点数
        W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))  # 初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
        b1 = tf.Variable(tf.zeros([h1_units]))  # 隐含层偏置b1全部初始化为0
        W2 = tf.Variable(tf.zeros([h1_units, 10]))
        b2 = tf.Variable(tf.zeros([10]))
        x = tf.placeholder(tf.float32, [None, in_units])
        keep_prob = tf.placeholder(tf.float32)  # Dropout失活率

        # 定义模型结构
        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        hidden1_drop = tf.nn.dropout(hidden1, rate=1 - keep_prob)
        y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

        self.add_inputs(x=x, keep_prob=keep_prob)
        self.add_outputs(y)
        return self


class MLPCompile(TFCompile):

    def compile(self):
        # 训练部分
        y = self.outputs_list()[0]
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
        train_op = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', cross_entropy)

        self.add_inputs(y=y_)
        self.add_metrics(train_op=train_op, loss=cross_entropy, accuracy=accuracy)
        return self
