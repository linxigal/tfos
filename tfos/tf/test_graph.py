#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/10 14:15
:File   :test_graph.py
:content:
  
"""

import sys
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import data_flow_ops


def build_graph():
    with tf.device("cpu:0"):
        a = tf.placeholder(tf.float32, shape=[1, ], name='a')
        # print(a.name)
        # b = tf.Variable(tf.truncated_normal([10, 1], name='b'), name='b')
        # b = tf.truncated_normal([10, 1], name='b')
        # b = tf.Variable(tf.constant(3, tf.float32, name='b'), name='b')
        # b = tf.Variable(tf.constant(np.random.random(), tf.float32), name='b')
        b = tf.Variable(tf.random.normal([1, ]), name='b')
        b_1 = tf.Variable(tf.random.normal([1, ]), name='b_1')
        # c = tf.constant(5, tf.float32, name='w')
        c = tf.random.normal([1, ], name='c')
        d = tf.add(tf.multiply(a, b, name='mul'), c, name='d')
        # d = tf.add(a, c, name='add')
        tf.add_to_collection('a', a)
        tf.add_to_collection("d", d)
        index_queue = tf.train.range_input_producer(32, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32, name='index_queue')
        tf.add_to_collection('index', index_queue)
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        tf.add_to_collection("test", image_paths_placeholder)
        tf.add_to_collection("test", labels_placeholder)
        tf.add_to_collection("test", control_placeholder)
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                              dtypes=[tf.string, tf.int32, tf.int32],
                                              shapes=[(1,), (1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many(
            [image_paths_placeholder, labels_placeholder, control_placeholder],
            name='enqueue_op')
        tf.add_to_collection('op', enqueue_op)


def save_graph(graph_name):
    build_graph()
    saver = tf.train.Saver()

    if flag == 1:
        print(saver.export_meta_graph(graph_name, as_text=True, clear_devices=True))
    elif flag == 2:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, graph_name)
    elif flag == 3:
        graph = tf.get_default_graph()
        # print(graph.as_graph_def().SerializeToString())
        # tf.train.write_graph(tf.get_default_graph(), '.', 'ttt.pbtxt')
        print("*" * 100)
        # print(tf.train.export_meta_graph(graph=graph).SerializeToString())
        print(tf.train.export_meta_graph(graph=graph, as_text=True))


def serialize_graph():
    build_graph()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     tf.train.write_graph(sess.graph_def, ".", "test.pb", as_text=False)
    # graph_str = tf.get_default_graph().as_graph_def().SerializeToString()
    graph = tf.get_default_graph()
    graph_str = tf.train.export_meta_graph(graph=graph).SerializeToString()
    # graph_str = tf.train.export_meta_graph(graph=graph, as_text=True)
    graph_str = bytes(graph_str.decode('latin'), encoding='latin')
    # print(graph_str)
    # print("*" * 100)
    # print(bytes(graph_str.decode('latin'), encoding='utf8'))
    # print(graph_str.decode('ASCII'))
    with tf.gfile.GFile("test.meta", 'wb') as f:
        f.write(graph_str)
    # return graph_str


def deserialize_graph():
    # graph_def = tf.GraphDef()
    # graph_def.ParseFromString(graph_str)

    model_filename = "test.meta"
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        # graph_def = tf.GraphDef()
        # graph_def.ParseFromString(f.read())
        meta_graph = tf.MetaGraphDef()
        meta_graph.ParseFromString(f.read())

    # tf.import_graph_def(graph_def)
    tf.train.import_meta_graph(meta_graph)

    print(tf.train.export_meta_graph(graph=tf.get_default_graph()))


def load_graph(filename):
    with tf.device("cpu:0"):
        saver = tf.train.import_meta_graph(filename)
        print(tf.trainable_variables())
        print(tf.get_collection("a"))
        print(tf.get_collection("d"))
        # graph = tf.get_default_graph()
        # print(graph.get_tensor_by_name("a:0"))
        # print(tf.global_variables())
        print("*" * 100)

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
        print(tf.get_default_graph().as_graph_def())
        # return saver


def exe_graph():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # a = tf.get_collection("a")[0]
        a = tf.get_default_graph().get_tensor_by_name("a:0")
        d = tf.get_default_graph().get_tensor_by_name("d:0")
        # print(tf.get_default_graph().get_operation_by_name('enqueue_op'))
        # print(tf.get_default_graph().get_operation_by_name('index_queue'))
        # d = tf.get_collection("d")[0]
        result = sess.run([d], feed_dict={a: [2]})
        print(result)
        # print(a.name)
        # print(d.name)
        print(tf.get_collection('test'))


if __name__ == '__main__':
    flag = 3
    name = "test_graph_{}.meta".format(flag)
    # build_graph()
    # save_graph(name)
    # tf.reset_default_graph()
    # load_graph(name)
    serialize_graph()
    tf.reset_default_graph()
    deserialize_graph()
    exe_graph()
