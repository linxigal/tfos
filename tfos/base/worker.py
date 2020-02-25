# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os


class Worker(object):
    def __init__(self, batch_size=1,
                 epochs=1,
                 steps_per_epoch=1,
                 save_dir=None,
                 result_dir=None,
                 checkpoint_dir=None,
                 log_dir=None,
                 name='model'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.name = name
        self.task_index = None
        self.job_name = None
        self.tf_feed = None
        self.cluster = None
        self.server = None
        self.model = None
        self.model_suffix = None

    @property
    def checkpoint_path(self):
        if self.model_suffix == 'h5':
            filename = self.name + '-{epoch}'
        elif self.model_suffix == 'pb':
            filename = self.name
        else:
            raise ValueError("save model suffix couldn't be None, choices pb|h5")
        return os.path.join(self.checkpoint_dir, filename)

    @property
    def model_path(self):
        if self.model_suffix is None:
            raise ValueError("save model suffix couldn't be None, choices pb|h5")
        return os.path.join(self.save_dir, '{}.{}'.format(self.name, self.model_suffix))
