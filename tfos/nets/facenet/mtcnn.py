#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/12/31 11:06
:File   :mtcnn.py
:content:
  
"""
import os
import random
import time

import numpy as np
import tensorflow as tf
from facenet.src import facenet
from facenet.src.align import detect_face
# from scipy import misc
from imageio import imread, imsave
from skimage.transform import resize
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflowonspark import TFCluster
from tensorflowonspark import TFNode

from tfos.base import ext_exception


class TFOS_MTCNN(object):
    def __init__(self, sc, sqlc=None, cluster_size=2, num_ps=1, input_mode=TFCluster.InputMode.SPARK):
        self.sc = sc
        self.sqlc = sqlc
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.input_mode = input_mode
        self.cluster = None
        self.tf_args = {}

    @property
    def num_workers(self):
        num_workers = self.cluster_size - self.num_ps
        assert num_workers > 0, "cluster_size, num_ps must be positive, and cluster_size > num_ps"
        return num_workers

    @ext_exception("mtcnn process")
    def run(self, input_dir, output_dir, *args, **kwargs):
        out_text_dir = os.path.join(output_dir, 'text')
        out_image_dir = os.path.join(output_dir, 'images')
        out_result_dir = os.path.join(output_dir, 'result')
        if tf.io.gfile.exists(out_text_dir):
            tf.io.gfile.rmtree(out_text_dir)
        if tf.io.gfile.exists(out_image_dir):
            tf.io.gfile.rmtree(out_image_dir)
        tf.io.gfile.makedirs(out_text_dir)
        tf.io.gfile.makedirs(out_image_dir)
        tf.io.gfile.makedirs(out_result_dir)
        dataset = facenet.get_dataset(input_dir)
        data_rdd = self.sc.parallelize([(cls.name, cls.image_paths) for cls in dataset])
        worker = MTCNNWorker(out_text_dir, out_image_dir, out_result_dir, *args, **kwargs)
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd, feed_timeout=60000)
        cluster.shutdown()


class MTCNNWorker(object):

    def __init__(self, out_text_dir, out_image_dir, out_result_dir, image_size=188, margin=44,
                 random_order=True, gpu_memory_fraction=1.0, detect_multiple_faces=False):
        self.out_text_dir = out_text_dir
        self.out_image_dir = out_image_dir
        self.out_result_dir = out_result_dir
        self.image_size = image_size
        self.margin = margin
        self.random_order = random_order
        self.gpu_memory_fraction = gpu_memory_fraction
        self.detect_multiple_faces = detect_multiple_faces
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.task_index = None
        self.job_name = None
        self.tf_feed = None
        self.cluster = None
        self.server = None
        self.tmp_dir = "/tmp/tfos/{}".format(int(time.time() * 1000))
        self.pnet, self.rnet, self.onet = [None] * 3

    def create_tmp_dir(self):
        tf.io.gfile.makedirs(self.tmp_dir)

    def delete_tmp_dir(self):
        if tf.io.gfile.exists(self.tmp_dir):
            tf.io.gfile.rmtree(self.tmp_dir)

    def generate_rdd_data(self):
        while not self.tf_feed.should_stop():
            batches = self.tf_feed.next_batch(1)
            if not batches:
                raise StopIteration()
            yield batches[0]

    def process_data(self, class_name, image_paths, text_file, result_file):
        output_class_dir = os.path.join(self.out_image_dir, class_name)
        tf.io.gfile.mkdir(output_class_dir)
        if self.random_order:
            random.shuffle(image_paths)
        for image_path in image_paths:
            file = os.path.split(image_path)[1]
            filename = os.path.splitext(file)[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            tmp_path = os.path.join(self.tmp_dir, file)
            tf.io.gfile.copy(image_path, tmp_path, True)

            try:
                # img = misc.imread(tmp_path)
                img = imread(tmp_path)
            except (IOError, ValueError, IndexError) as e:
                error_message = '{}: {}\n'.format(image_path, e)
                result_file.write(error_message)
            else:
                if img.ndim < 2:
                    result_file.write('Unable to align "%s"' % image_path)
                    text_file.write('%s\n' % (output_filename))
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]

                bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                            self.threshold, self.factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces > 1:
                        if self.detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            index = np.argmax(
                                bounding_box_size - offset_dist_squared * 2.0)
                            det_arr.append(det[index, :])
                    else:
                        det_arr.append(np.squeeze(det))

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                        bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                        bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        # scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                        scaled = np.array(Image.fromarray(cropped).resize((self.image_size, self.image_size),
                                                                          resample=Image.BILINEAR))
                        filename_base, file_extension = os.path.splitext(tmp_path)
                        if self.detect_multiple_faces:
                            tmp_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                        else:
                            tmp_filename_n = "{}{}".format(filename_base, file_extension)
                        # misc.imsave(tmp_filename_n, scaled)
                        imsave(tmp_filename_n, scaled)
                        output_filename_n = os.path.join(output_class_dir, os.path.split(tmp_filename_n)[1])
                        tf.io.gfile.copy(tmp_filename_n, output_filename_n, True)
                        text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                else:
                    result_file.write('Unable to align "%s"' % image_path)
                    text_file.write('%s\n' % (output_filename))

    def process(self):
        bounding_boxes_filename = os.path.join(self.out_text_dir, 'bounding_boxes_{}.txt'.format(self.task_index))
        result_filename = os.path.join(self.out_result_dir, 'result_{}.txt'.format(self.task_index))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        with tf.Session(self.server.target, config=config) as sess:
            K.set_session(sess)
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
            with tf.io.gfile.GFile(bounding_boxes_filename, "w") as text_file:
                with tf.io.gfile.GFile(result_filename, 'w') as result_file:
                    for class_name, image_paths in self.generate_rdd_data():
                        self.process_data(class_name, image_paths, text_file, result_file)

    def __call__(self, args, ctx):
        self.task_index = ctx.task_index
        self.job_name = ctx.job_name
        self.cluster, self.server = TFNode.start_cluster_server(ctx)
        self.tf_feed = TFNode.DataFeed(ctx.mgr)
        if ctx.job_name == "ps":
            self.server.join()
        elif ctx.job_name == "worker":
            self.create_tmp_dir()
            self.process()
            self.delete_tmp_dir()
