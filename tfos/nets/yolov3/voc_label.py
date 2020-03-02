#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/26 9:07
:File   :voc_label.py
:content:
  
"""

import xml.etree.ElementTree as ET
from os.path import join

import tensorflow as tf


class VOCLabelLayer(object):
    data_sets = ['train', 'val', 'test']
    classes_filename = 'voc_classes.txt'

    def __init__(self, data_dir, output_dir, image_format='jpg'):
        self.data_dir = data_dir
        self.image_dir = join(self.data_dir, 'JPEGImages')
        self.annotations_dir = join(self.data_dir, 'Annotations')
        self.output_dir = output_dir
        self.image_format = image_format
        class_file = join(self.output_dir, self.classes_filename)
        self.classes = self.get_classes(class_file)

    @staticmethod
    def get_classes(class_file):
        with tf.io.gfile.GFile(class_file) as f:
            return f.read().strip().split()

    def do(self):
        for image_set in self.data_sets:
            image_ids_path = join(self.data_dir, 'ImageSets/Main/{}.txt'.format(image_set))
            if not tf.io.gfile.exists(image_ids_path):
                continue
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
                if cls not in self.classes or int(difficult) == 1:
                    continue
                cls_id = self.classes.index(cls)
                xml_box = obj.find('bndbox')
                b = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                     int(xml_box.find('ymax').text))
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
