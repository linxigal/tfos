#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/31 16:00
:File       : lstm.py
"""

from tfos.contrib.neure.lstm import lstm_cell
from tfos.tfos import TFOSLocal
from tensorflow.examples.tutorials.mnist import input_data
from examples import *

mnist = input_data.read_data_sets("./TensorFlowOnSpark/mnist", one_hot=True)


def build_model():
    pass

def main(unused_args):
    pass


if __name__ == "__main__":
    tf.app.run()
