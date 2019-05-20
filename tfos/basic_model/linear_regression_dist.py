# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 22:49
:File    : logistic_regression_dist.py
"""

import logging
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf


def map_fun(args, ctx):
    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    # Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args['rdma'])

    # Create generator for Spark data feed
    tf_feed = ctx.get_data_feed(args['mode'] == 'train')

    def rdd_generator():
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)
            if len(batch) == 0:
                return
            row = batch[0]
            data = np.array(row[0]).astype(np.float32)
            label = np.array(row[1]).astype(np.int64)
            yield (data, label)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(
                tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):

            # Dataset for input data
            ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32),
                                                (tf.TensorShape([4]), tf.TensorShape([1]))).batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()
            x, y_ = iterator.get_next()

            # Set model weights
            W = tf.Variable(tf.truncated_normal([4, 3], stddev=1), name="weight")
            b = tf.Variable(tf.zeros([3]), name="bias")
            tf.summary.histogram("hidden_weights", W)

            global_step = tf.train.get_or_create_global_step()

            # Construct a linear model
            pred = tf.add(tf.multiply(x, W), b)

            # Mean squared error
            n_samples = x.shape[0]
            loss = tf.reduce_sum(tf.pow(pred - y_, 2)) / (2 * n_samples)

            tf.summary.scalar("loss", loss)
            # Gradient Descent
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = ctx.absolute_path(args.model_path)
        print("tensorflow model path: {0}".format(logdir))
        summary_writer = tf.summary.FileWriter(os.path.join(args.tb_path, "tensorboard_%d" % worker_num),
                                               graph=tf.get_default_graph())

        hooks = [tf.train.StopAtStepHook(last_step=args.steps)] if args.mode == "train" else []
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op,
                                                                          saver=saver),
                                               checkpoint_dir=logdir,
                                               hooks=hooks) as sess:
            print("{} session ready".format(datetime.now().isoformat()))
            # Loop until the session shuts down or feed has no more data
            step = 0
            while not sess.should_stop() and not tf_feed.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                if args.mode == "train":
                    _, summary, step = sess.run([train_op, summary_op, global_step])
                    if (step % 2 == 0) and (not sess.should_stop()):
                        # print("{} step: {} accuracy: {}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
                        logging.error(
                            "{} step: {} accuracy: {}".format(datetime.now().isoformat(), step, sess.run(loss)))
                    if task_index == 0:
                        summary_writer.add_summary(summary, step)
                else:  # args.mode == "inference"
                    y_s, preds = sess.run([y_, pred])
                    results = ["{} real: {}, Prediction: {}".format(datetime.now().isoformat(), l, p) for l, p in
                               zip(y_s, preds)]
                    tf_feed.batch_results(results)

        print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

        if sess.should_stop() or step >= args.steps:
            tf_feed.terminate()

            # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
            # wait for all other nodes to complete (via done files)
            done_dir = "{}/{}/done".format(ctx.absolute_path(args.model_path), args.mode)
            print("Writing done file to: {}".format(done_dir))
            tf.gfile.MakeDirs(done_dir)
            with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
                done_file.write("done")

            for i in range(60):
                if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
                    print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
                    time.sleep(1)
                else:
                    print("{} All nodes done".format(datetime.now().isoformat()))
                    break
