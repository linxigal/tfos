#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/26 9:07
:File   :voc_label.py
:content:
  
"""

import tensorflow as tf
import xml.etree.ElementTree as ET

from os.path import join

sets = ['train', 'val']

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class VOCLabelLayer(object):

    def __init__(self, data_dir, output_dir, image_format='jpg'):
        self.data_dir = data_dir
        self.image_dir = join(self.data_dir, 'JPEGImages')
        self.annotations_dir = join(self.data_dir, 'Annotations')
        self.output_dir = output_dir
        self.image_format = image_format

    def do(self):
        for image_set in sets:
            image_ids_path = join(self.data_dir, 'ImageSets/Main/{}.txt'.format(image_set))
            output_path = join(self.output_dir, '{}.txt'.format(image_set))
            with tf.io.gfile.GFile(image_ids_path) as image_ids_file:
                image_ids = image_ids_file.read().strip().split()
                with tf.io.gfile.GFile(output_path, 'w') as output:
                    for image_id in image_ids:
                        output.write(join(self.image_dir, "{}.{}".format(image_id, self.image_format)))
                        self.convert_annotation(image_id, output)
                        output.write('\n')

    def convert_annotation(self, image_id, list_file):
        with tf.io.gfile.GFile(join(self.annotations_dir, "{}.xml".format(image_id))) as in_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                     int(xmlbox.find('ymax').text))
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
