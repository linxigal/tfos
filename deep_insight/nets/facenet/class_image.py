#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/14 16:17
:File   :class_image.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class ClassImage(Base):
    def __init__(self, data_dir, image_size, valid_ratio=0.0, min_images_per_class=0, mode="SPLIT_IMAGES",
                 random_rotate=True, random_crop=True, random_flip=True, use_fixed_image_standardization=True):
        super(ClassImage, self).__init__()
        self.p('data_dir', data_dir)
        self.p('image_size', image_size)
        self.p('valid_ratio', valid_ratio)
        self.p('min_images_per_class', min_images_per_class)
        self.p('random_rotate', random_rotate)
        self.p('random_crop', random_crop)
        self.p('random_flip', random_flip)
        self.p('use_fixed_image_standardization', use_fixed_image_standardization)

    def run(self):
        param = self.params
        from tfos.utils import convert_bool
        from tfos.choices import BOOLEAN
        from tfos.nets.facenet.class_image import ClassImage

        # param = json.loads('<#zzjzParam#>')
        data_dir = param.get('data_dir')
        image_size = param.get('image_size')
        valid_ratio = param.get('valid_ratio', '0.0')
        min_images_per_class = param.get('min_images_per_class', '0')
        mode = param.get('mode')
        random_rotate = param.get('random_rotate', BOOLEAN[0])
        random_crop = param.get('random_crop', BOOLEAN[0])
        random_flip = param.get('random_flip', BOOLEAN[0])
        use_fixed_image_standardization = param.get('use_fixed_image_standardization', BOOLEAN[0])

        kwargs = dict(sc=sc, data_dir=data_dir, image_size=int(image_size))

        if valid_ratio:
            kwargs['valid_ratio'] = float(valid_ratio)
        if min_images_per_class:
            kwargs['min_images_per_class'] = int(min_images_per_class)
        if mode:
            kwargs['mode'] = mode
        if random_rotate:
            kwargs['random_rotate'] = convert_bool(random_rotate)
        if random_crop:
            kwargs['random_crop'] = convert_bool(random_crop)
        if random_flip:
            kwargs['random_flip'] = convert_bool(random_flip)
        if use_fixed_image_standardization:
            kwargs['use_fixed_image_standardization'] = convert_bool(use_fixed_image_standardization)

        train_df, val_df = ClassImage(**kwargs).process_data()
        train_df.persist()
        val_df.persist()
        train_df.show()
        val_df.show()
        outputRDD('<#zzjzRddName#>_train_data', train_df)
        outputRDD('<#zzjzRddName#>_val_data', val_df)


class TestClassImage(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        path = ROOT_PATH if self.is_local else HDFS
        self.data_dir = os.path.join(path, 'data/data/tmp')

    def test_class_image(self):
        ClassImage(self.data_dir, image_size='160').run()


if __name__ == '__main__':
    unittest.main()
