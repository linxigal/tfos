#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/12 16:27
:File       : make_regreesion.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

current = os.path.dirname(__file__)
root_path = os.path.dirname(os.path.dirname(current))

x, y = make_regression(10, 5, noise=0.1)

# data = np.c_[x, y.reshape(10, 1)]
data = np.hstack((x, y.reshape(10, 1)))
save_path = os.path.join(root_path, "output_data", 'data', 'regression_data.csv')
# np.savetxt(save_path, data, delimiter=",")
pd.DataFrame(data, columns=range(6)).to_csv(save_path)
