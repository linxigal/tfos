#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 15:34
:File       : df2inputs.py
"""

from deep_insight.base import *


class DF2Inputs(Base):
    def __init__(self, input_table_name, target_field):
        super(DF2Inputs, self).__init__()
        self.p('input_table_name', input_table_name)
        self.p('target_field', target_field)

    def run(self):
        param = self.params

        def row2list(row):
            row_dict = row.asDict()
            label = row_dict.pop(target_field)
            features = [value for key, value in sorted(row_dict.items(), key=lambda k: k[0])]
            return features, label

        # param = json.loads('<#zzjzParam#>')
        input_table_name = param.get('input_table_name')
        target_field = param.get('target_field')
        input_rdd = inputRDD(input_table_name)

        output_rdd = input_rdd.map(row2list).toDF(['features', 'label'])
        outputRDD('<#zzjzRddName#>_data', output_rdd)


if __name__ == "__main__":
    import os
    from deep_insight import ROOT_PATH
    from deep_insight.data.read_csv import ReadCsv

    filepath = os.path.join(ROOT_PATH, 'output_data', 'data', 'regression_data.csv')
    ReadCsv(filepath).run()
    DF2Inputs('<#zzjzRddName#>', '5').run()
