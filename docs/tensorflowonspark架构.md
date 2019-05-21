# TensorFlowOnSpark架构

## 1.架构分析
![架构分析](images/tfos架构分析)

***TF core是什么？***
***为什么HDFS只和worker相连而与PS无关？***

TensorFlowOnSpark的架构较为简单，Spark Driver程序并不会参与TensorFlow内部相关的计算和处理。其设计思路像是将一个TensorFlow集群运行在了Spark上，其在每个Spark Executor中启动TensorFlow应用程序，然后通过gRPC或RDMA方式进行数据传递与交互。


## 2. 生命周期
![生命周期](images/tfos生命周期)

TensorFlowOnSpark的Spark应用程序包括4个基本过程。
1. Reserve：组建TensorFlow集群，并在每个Executor进程上预留监听端口，启动“数据/控制”消息的监听程序。
2. Start：在每个Executor进程上启动TensorFlow应用程序；
3. Train/Inference：在TensorFlow集群上完成模型的训练或推理
4. Shutdown：关闭Executor进程上的TensorFlow应用程序，释放相应的系统资源(消息队列)。

![作业提交](images/tfos作业提交)

用户直接通过spark-submit的方式提交Spark应用程序(mnist_spark.py)。其中通过--py-files选项附带TensorFlowOnSpark框架(tfspark.zip)，及其TensorFlow应用程序(mnist_dist.py)，从而实现TensorFlow集群在Spark平台上的部署。

![TensorFlow集群：数据集视图](images/tfos数据集视图)

首先看看TensorFlow集群的建立过程。

首先根据spark-submit传递的num_executor参数，通过调用cluster = sc.parallelize(num_executor)建立一个ParllelCollectionRDD，其中分区数为num_executor。也就是说，此时分区数等于Executor数。

然后再调用cluster.mapPartitions(TFSparkNode.reserve)将ParllelCollectionRDD变换(transformation)为MapPartitionsRDD，在每个分区上回调TFSparkNode.reserve。

TFSparkNode.reserve将会在该节点上预留一个端口，并驻留一个Manager服务。Manager持有一个队列，用于完成进程间的同步，实现该节点的“数据/控制”消息的服务。

数据消息启动了两个队列：Input与Output，分别用于RDD与Executor进程之间的数据交换。

控制消息启动了一个队列：Control，用于Driver进程控制PS任务的生命周期，当模型训练完成之后，通过Driver发送Stop的控制消息结束PS任务。

![TensorFlow集群：任务集视图](images/tfos任务集视图)

这是从分区的角度看待TensorFlow集群建立的过程，横轴表示RDD。这里存在两个RDD，第一个为ParllelCollectionRDD，然后变换为MapPartitionsRDD。

纵轴表示同一个分区(Partition)，并在每个分区上启动一个Executor进程 。在Spark中，分区数等于最终在TaskScheduler上调度的Task数目。

此处，sc.parallelize(num_executor)生成一个分区数为num_executor的ParllelCollectionRDD。也就是说，此时分区数等于num_executor数目。

在本例中，num_executor为3，包括1个PS任务，2个Worker任务。


![TensorFlow集群：领域模型](images/tfos领域模型)

TensorFlow集群建立后，将生成上图所示的领域模型。其中，一个TFCluster将持有num_executor个TFSparkNode节点；在每个TFSparkNode上驻留一个Manager服务，并预留一个监听端口，用于监听“数据/控制”消息。

实际上，TFSparkNode节点承载于Spark Executor进程之上。


## 3. 启动

![启动过程](images/tfos启动过程)
TensorFlow集群建立后，通过调用cluster.start启动集群服务。其结果将在每个Executor进程上启动TensorFlow应用程序。
此处，需要对原生的TensorFlow应用程序进行适配修改，包括2个部分：
Feeding与Fetching: 数据输入/输出机制修改
ClusterSpec: TF集群的构造描述
其余代码都将保留，最小化TensorFlow应用程序的修改。

![启动过程2](images/tfos启动过程2)

在cluster上调用foreachPartition(TFSparkNode.start(map_func))，将在每个分区(Executor进程)上回调TFSparkNode.start(map_func)。其中，map_func是对应TF应用程序的包装。

通过上述过程，在Spark上拉起了一个TF的集群服务。从而使得Spark集群拥有了深度学习和GPU加速的能力。


## 4.数据供给

1. TensorFlow QueueRunner: FileReader & QueueRunner
2. Spark Feeding: RDD->Executor->TensorFlow Graph

当Spark平台上已经拉起了TF集群服务之后，便可以启动模型的训练或推理过程了。在训练或推理过程中，最重要的是解决数据的Feeding和Fetching问题。

TFoS上提供了两种方案：

TensorFlow QueueRunner：利用TensorFlow提供的FileReader和QueueRunner机制。Spark未参与任何工作，请查阅TensorFlow官方相关文档。

Spark Feeding：首先从RDD读取分区数据(通过HadoopRDD.compute)，然后将其放在Input队列中，Executor进程再从该队列中取出，并进一步通过feed_dict，调用session.run将分区数据供给给TensorFlow Graph中。

![Spark Feeding: Input Queue](images/tfosFeedingInputQueue)

Feeding过程，就是通过Input Queue同步实现的。当RDD读取分区数据后，阻塞式地将分区数据put到Input队列中；TFGraph在session.run获取Next Batch时，也是阻塞式地等待数据的到来。

![Spark Fetching: Output Queue](images/tfosFetchingOutputQueue)

同样的道理，Fetching过程与Feeding过程类同，只是使用Output Queue，并且数据流方向相反。

session.run返回的数据，通过put阻塞式地放入Output Queue，RDD也是阻塞式地等待数据到来。

![模型训练](images/tfos模型训练)

以模型训练过程为例，讲解RDD的变换过程。此处以Mnist手写识别为例，左边表示X，右边表示Y。分别通过HadoopRDD读取分区数据，然后通过MapPartititionRDD变换分区的数据格式；然后通过zip算子，实现两个RDD的折叠，生成ZipPartitionsRDD。

然后，根据Epoches超参数的配置，将该RDD重复执行Epoches次，最终将结果汇总，生成UnionRDD。

在此之前，都是Transformation的过程，最终调用foreachPartition(train)启动Action，触发Spark Job的提交和任务的运行。

## 5.关闭队列

![关闭队列](images/tfos关闭队列)

当模型训练或推理完成之后，分别在Input/Control队列中投掷Stop(以传递None实现)消息，当Manager收到Stop消息后，停止队列的运行。

最终，Spark应用程序退出，Executor进程退出，整个工作流执行结束。

## 参考
[TensorFlowOnSpark架构设计](https://www.jianshu.com/p/8ab0718f6967)
[TensorFlow遇上Spark](https://www.jianshu.com/p/62b4ebb5a2f4#)