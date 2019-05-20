# TensorFlowOnSpark 环境配置

### java环境

1. JDK [下载链接](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

2. 环境变量设置
    ```
    export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-i386/
    export PATH=${JAVA_HOME}/bin:$PATH
    ```

### scala环境
1. [下载链接](https://www.scala-lang.org/download/)
2. 环境变量设置
    ```
    export SCALA_HOME=/Users/wjl/tools/scala-2.12.8
    export PATH=$PATH:$SCALA_HOME/bin
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    ```

### hadoop环境
1. [下载链接](https://hadoop.apache.org/releases.html)
2. 环境变量设置
    ```
    export HADOOP_HOME=/Users/wjl/tools/hadoop-2.8.5
    export PATH=${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:$PATH
    export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
    export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib"
    ```


### spark环境
1. [下载链接](https://spark.apache.org/downloads.html)
2. 环境变量设置
    ```
    # 系统变量
    export SPARK_HOME="/Users/wjl/tools/spark-2.4.3-bin-hadoop2.7"
    export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
    export PYSPARK_PYTHON=/Users/wjl/.virtualenvs/tensorflow/bin/python
    export PYSPARK_DRIVER_PYTHON=python3
    export PYSPARK_DRIVER_PYTHON_OPTS=""
    export PYTHONPATH="/Users/wjl/tools/spark-2.4.3-bin-hadoop2.7/python"
    
    # 集群设置（Launch standalone Spark cluster）
    #export MASTER=spark://IP:7077
    export SPARK_WORKER_INSTANCES=2
    export CORES_PER_WORKER=1
    export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
    
    # 集群启动
    ${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G ${MASTER}
    
    # 项目设置
    export TFoS_HOME=/Users/wjl/github/TensorFlowOnSpark
    ```