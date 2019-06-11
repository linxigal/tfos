#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 15:34
:File       : test_df2data.py
"""


from examples.base import *


class DF2Data(Base):
    def __init__(self, input_table_name, target_field):
        super(DF2Data, self).__init__()
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
        outputRDD('<#zzjzRddName#>', output_rdd)


if __name__ == "__main__":
    DF2Data()
