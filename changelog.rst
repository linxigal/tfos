Changelog
=========


0.2.3    ( 2019-09-24 )
-----------------------

New
~~~
- Add alexnet network model. [wwwa]

Fix
~~~
- Debug model train,evaluate,predict. [wwwa]

  - rebuild mnist dataset
  - debug mnist mlp model
  - debug lenet network model

  new: debug evaluate and predict


0.2.3 (2019-09-18)
------------------

New
~~~
- Add recurrent layers and rebuild train model. [wwwa]

  - add recurrent.py
  - add model evaluate and predict

Fix
~~~
- Fix compile optimizer layer. [wwwa]


0.2.1 (2019-09-10)
------------------

New
~~~
- Rebuild neural network struct. [wwwa]

  - add network Model
  - rebuild train and inference operator
  - fix core layer operators
  - fix convolution layer operators
  - fix pooling layer operators
  - add merge layer operator


0.1.1 (2019-09-05)
------------------

New
~~~
- Test train and inference model. [wwwa]
- Add core layers for core.py. [wwwa]
- Add operator annotation. [wwwa]
- Add pooling.py. [wwwa]

  - add MaxPool1D, MaxPool2D, MaxPool3D, AvgPool1D, AvgPool2D, AvgPool3D layer
- Add activation.py. [wwwa]

  - add Activation, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU
- Add convolution.py. [wwwa]

  - add Convolution1D,Convolution2D,Convolution3D
  - add Convolution2DTranspose, Convolution3DTranspose
  - add SummaryLayer
- Add file_manager.py. [wwwa]
- 增加gitchangelog配置. [wwwa]

Changes
~~~~~~~
- Add mnist cgan mlp operator. [wwwa]

Fix
~~~
- Delete redundancy code. [wwwa]
- Core.py add annotation. [wwwa]
- Cgan network matplot savefig to hdfs. [wwwa]
- Fix operator couldn't execute on deepinsight. [wwwa]

Other
~~~~~
- Test: train model and inference model. [wwwa]
- Docs: add annotation for convolution. [wwwa]
- Add gan network. [wwwa]


0.1.0 (2019-07-26)
------------------

New
~~~
- 增加gitchangelog配置. [wwwa]
- Add .gitchangelog.rc. [wwwa]
- Add test_inference.py. [wwwa]
- Add test_keras_mnist.py. [wwwa]
- Add flatten,convolution,drop,embedding,lstm,max_pooling etc layer.
  [wwwa]
- Change test_train.py. [wwwa]
- Train suanzi. [wwwa]
- Dropout layer and train data type convert. [wwwa]

Changes
~~~~~~~
- Rename examples to deep_insight. [wwwa]
- Package code. [wwwa]
- Fabfile.py add reinstall function. [wwwa]
- Train model result save to h5. [wwwa]
- Test_inference.py. [wwwa]
- Add test_inference.py. [wwwa]
- Add test_inference.py. [wwwa]
- Restore model for model.load_weights() [wwwa]
- Add train model checkpoint. [wwwa]
- Model train relative. [wwwa]
- Add lstm,cnn,multilayer_perceptron. [wwwa]
- Test fabfile. [wwwa]
- Python_env.md. [wwwa]

Fix
~~~
- Add test_optimzer.py and add test_read_mnist.py. [wwwa]
- Test_train.py. [wwwa]

Other
~~~~~
- Debug train and inference. [wwwa]
- Test: add test_keras_model_save. [wwwa]
- Docs: change python_env.md. [wwwa]
- Add fabfile.py and setup.py. [wwwa]
- Add image2tfrecord and perceptron. [wwwa]
- Add read csv file. [wwwa]
- Add tensorflow monitore and checkpoint document. [wwwa]
- Tfos model initialize. [wwwa]
- Build execute graph. [wwwa]
- Packaging TFOS class. [wwwa]
- New regression.py. [wwwa]
- Rebuild project directory. [wwwa]
- Rebuild project directory. [wwwa]
- Modify TensorFlowDatasetAPI.md. [wwwa]
- Modify TensorFlowDatasetAPI.md. [wwwa]
- Add tensorflowonspark架构.md. [wwwa]
- 增加TensorFlowDatasetAPI.md. [wwwa]
- 添加python_env.md文档. [wwwa]
- 修改tensorboard配置路径，添加env.md环境设置文件. [wwwa]
- 增加线性回归代码. [wwwa]
- 将utils目录移动到tfos目录下. [wwwa]
- 添加docs文档目录. [wwwa]
- 去掉对output_data文件夹的跟踪. [wwwa]
- 初始化项目，并实现逻辑归回模型. [wwwa]
- Initial commit. [jinlong]


