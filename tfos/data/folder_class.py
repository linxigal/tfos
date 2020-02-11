# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import math
# from facenet.src import facenet
import os

import numpy as np

from tfos.data import DATA_INDEX, MARK_INDEX

SPLIT_MODE = ['SPLIT_IMAGES', 'SPLIT_CLASSES']


class ImageClass(object):
    """Stores the paths to images for a given class"""

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_data_set(path, has_class_directories=True):
    data_set = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    n_classes = len(classes)
    for i in range(n_classes):
        class_name = classes[i]
        class_dir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(class_dir)
        data_set.append(ImageClass(class_name, image_paths))

    return data_set


def get_image_paths(class_dir):
    image_paths = []
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        image_paths = [os.path.join(class_dir, img) for img in images]
    return image_paths


def split_data_set(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode == 'SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes * (1 - split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode == 'SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class * (1 - split_ratio)))
            if split == nrof_images_in_class:
                split = nrof_images_in_class - 1
            if split >= min_nrof_images_per_class and nrof_images_in_class - split >= 1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def get_image_paths_and_labels(data_set):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(data_set)):
        image_paths_flat += data_set[i].image_paths
        labels_flat += [i] * len(data_set[i].image_paths)
    return image_paths_flat, labels_flat


class FolderClass(object):
    def __init__(self, sc, data_dir, split_ratio=0.0, min_num_per_class=1, mode='SPLIT_IMAGES'):
        self.__sc = sc
        self.__data_dir = data_dir
        self.__split_ratio = split_ratio
        self.__min_num_per_class = min_num_per_class
        assert mode in SPLIT_MODE
        self.__mode = mode
        self.__train_set, self.__test_set = self.init_data()

    def init_data(self):
        dataset = get_data_set(self.__data_dir)
        if self.__split_ratio > 0.0:
            train_set, test_set = split_data_set(dataset,
                                                 self.__split_ratio,
                                                 self.__min_num_per_class,
                                                 self.__mode)
        else:
            train_set, test_set = dataset, []
        return train_set, test_set

    @property
    def train_data(self):
        return self.process(self.__train_set)

    @property
    def test_data(self):
        return self.process(self.__test_set)

    def process(self, data_set):
        n_classes = len(data_set)
        image_list, label_list = get_image_paths_and_labels(data_set)
        data_rdd = self.__sc.parallelize(zip(image_list, label_list)).toDF(DATA_INDEX)
        mark_rdd = self.__sc.parallelize([(data_set[i].name, i) for i in range(len(data_set))]).toDF(MARK_INDEX)
        return n_classes, data_rdd, mark_rdd
