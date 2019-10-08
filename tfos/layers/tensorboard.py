#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/8 9:49
:File   :tensorboard.py
:content:
  
"""

import os
import socket
import subprocess
import tensorflow as tf

from tfos.base import BaseLayer, ext_exception


class TensorBoardLayer(BaseLayer):
    def __init__(self, log_dir, **kwargs):
        super(TensorBoardLayer, self).__init__(**kwargs)
        self.log_dir = log_dir

    @ext_exception("TensorBoard Layer")
    def add(self, operator):
        out = None
        if operator == 'start':
            self.start()
            out = self._add_or_create_column('url', '{}:6006'.format(self.get_ip()))
        elif operator == 'stop':
            self.stop()
        else:
            raise ValueError("TensorBoard operator incorrect!!!")
        return out

    def start(self):
        cmd = ['/usr/local/python3.6.4/bin/tensorboard', '--logdir', self.log_dir]
        subprocess.Popen(cmd)

    def stop(self):
        processes = self.find()
        if processes:
            cmd = "kill -9 {}".format(' '.join(processes))
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf8')

    @staticmethod
    def find(name='tensorboard'):
        cmd = "ps -ef|grep '%s'|grep -v grep|awk '{print $2}'" % name
        out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf8')
        infos = out.stdout.read().splitlines()
        return infos

    @staticmethod
    def get_ip():
        # try:
        #     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     s.connect(('8.8.8.8', 80))
        #     ip = s.getsockname()[0]
        # finally:
        #     s.close()
        # return ip
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
