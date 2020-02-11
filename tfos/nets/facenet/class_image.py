#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/14 15:07
:File   :class_image.py
:content:
  
"""

import tensorflow as tf
from facenet.src import facenet

from tfos.data import DATAINDEX


class ClassImage(object):

    def __init__(self, sc, data_dir, image_size, valid_ratio=0.0, min_images_per_class=0, mode="SPLIT_IMAGES",
                 random_rotate=True, random_crop=True, random_flip=True, use_fixed_image_standardization=True):
        self.sc = sc
        self.data_dir = data_dir
        self.image_size = (image_size, image_size)
        self.valid_ratio = valid_ratio
        self.min_images_per_class = min_images_per_class
        self.mode = mode
        self.random_rotate = random_rotate
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.use_fixed_image_standardization = use_fixed_image_standardization

    @property
    def control_value(self):
        _control_value = facenet.RANDOM_ROTATE * self.random_rotate + \
                         facenet.RANDOM_CROP * self.random_crop + \
                         facenet.RANDOM_FLIP * self.random_flip + \
                         facenet.FIXED_STANDARDIZATION * self.use_fixed_image_standardization
        return _control_value

    def process_data(self):
        dataset = facenet.get_dataset(self.data_dir)
        train_set, val_set = facenet.split_dataset(dataset, self.valid_ratio, self.min_images_per_class, self.mode)
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

        image_list = self.preprocess_image(image_list)
        val_image_list = self.preprocess_image(val_image_list)

        train_data = zip(image_list, label_list)
        val_data = zip(val_image_list, val_label_list)
        train_rdd = self.sc.parallelize(train_data)
        val_rdd = self.sc.parallelize(val_data)
        train_df = train_rdd.toDF(DATAINDEX)
        val_df = val_rdd.toDF(DATAINDEX)
        return train_df, val_df

    def preprocess_image(self, image_path_list):
        images = []
        for image_path in image_path_list:
            file_contents = tf.read_file(image_path)
            image = tf.image.decode_image(file_contents, 3)
            image = tf.cond(facenet.get_control_flag(self.control_value, facenet.RANDOM_ROTATE),
                            lambda: tf.py_func(facenet.random_rotate_image, [image], tf.uint8),
                            lambda: tf.identity(image))
            image = tf.cond(facenet.get_control_flag(self.control_value, facenet.RANDOM_CROP),
                            lambda: tf.random_crop(image, self.image_size + (3,)),
                            lambda: tf.image.resize_image_with_crop_or_pad(image, self.image_size[0],
                                                                           self.image_size[1]))
            image = tf.cond(facenet.get_control_flag(self.control_value, facenet.RANDOM_FLIP),
                            lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))
            image = tf.cond(facenet.get_control_flag(self.control_value, facenet.FIXED_STANDARDIZATION),
                            lambda: (tf.cast(image, tf.float32) - 127.5) / 128.0,
                            lambda: tf.image.per_image_standardization(image))
            image = tf.cond(facenet.get_control_flag(self.control_value, facenet.FLIP),
                            lambda: tf.image.flip_left_right(image),
                            lambda: tf.identity(image))
            image.set_shape(self.image_size + (3,))
            images.append(image)
        return images
