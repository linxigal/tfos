# tfos
TensorFlowOnSpark


## 简介

基于yahoo开源项目TensorFlowOnSpark项目，对TensorFlow和Spark很好的结合，可以实现对大数据进行分布式的机器学习以及深度网络学习。

## 项目结构

```markdown
- deep_insight  平台算子
    - base          平台基础包
    - compile       编译层
    - data          数据层
    - layers        算子层
    - model         模型训练层
    - nets          网络模型层
- docs          文档
- test          网络模型测试
    - test_lenet.py  LeNet网络模型
- tfos          框架打包
    - base          基础模块
    - compile       编译模块
    - layers        模型节点
    - data          数据处理
    - nets          网络模型
    - utils         常用工具方法
    - choices.py    可选参数模块
    - tfos.py       集群框架封装
    - worker.py     集群节点执行模块
```

## 安装

执行以下命令即可安装运行环境：
```
pip install -r requirements.txt
```

[python环境安装](docs/python_env.md)

[TensorFlowOnSpark环境搭建](docs/env.md) 

# 发布

[release](changelog.rst)
