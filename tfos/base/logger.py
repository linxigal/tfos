#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/9 17:47
:File   : logger.py
"""

import logging.handlers
from logging import Formatter

FORMATTER = "%(asctime)s - %(levelname)s - %(pathname)s - %(funcName)s - %(lineno)d : %(message)s"


class Logger(object):
    def __init__(self, name, level=logging.DEBUG, formatter=FORMATTER):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = Formatter(formatter)

    def sys_log(self, address=('127.0.0.1', 514)):
        handler = logging.handlers.SysLogHandler(address=address)
        handler.setFormatter(self.formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(sh)
        return self.logger


logger = Logger(__file__).sys_log(('192.168.21.27', 514))

if __name__ == '__main__':
    logger = Logger(__file__).sys_log()
    logger.debug("hello")
