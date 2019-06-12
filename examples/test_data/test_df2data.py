#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 15:34
:File       : test_df2data.py
"""

from examples.base import *


class TestDF2Inputs(Base):
    def __init__(self, input_table_name, target_field):
        super(TestDF2Inputs, self).__init__()
        self.p('input_table_name', input_table_name)
        self.p('target_field', target_field)

    def run(self):
        param = self.params

        def row2list(row):
            row_dict = row.asDict()
            target = row_dict.pop(target_field)
            data = [value for key, value in sorted(row_dict.items(), key=lambda k: k[1])]
            return data, target

        # param = json.loads('<#zzjzParam#>')
        input_table_name = param.get('input_table_name')
        target_field = param.get('target_field')
        input_rdd = inputRDD(input_table_name)

        output_rdd = input_rdd.map(row2list)
        print(output_rdd.take(2))
        outputRDD('<#zzjzRddName#>', output_rdd)


if __name__ == "__main__":
    TestDF2Inputs()
