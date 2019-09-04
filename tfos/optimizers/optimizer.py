#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/11 15:19
:File   : optimizer.py
"""
import json
from pyspark.sql.functions import lit
from tfos.base import *


class OptimizerLayer(BaseLayer):

    @ext_exception('optimizer layer')
    def add(self, loss, optimizer, metrics):
        if loss not in valid_losses:
            raise ValueError('model loss function incorrect!')
        if optimizer not in valid_optimizers:
            raise ValueError('model optimizer method incorrect!')

        check_metrics = []
        if metrics:
            if not isinstance(metrics, list):
                metrics = [metrics]
            for metric in metrics:
                if metric in valid_metrics:
                    check_metrics.append(metric)
                else:
                    raise ValueError(f"parameter metrics: {metric} is invalid!")

        optimizer_params = {
            'loss': loss,
            'optimizer': optimizer,
            'metrics': check_metrics if check_metrics else None
        }
        return self._add_column("compile_config", lit(json.dumps(optimizer_params)))
