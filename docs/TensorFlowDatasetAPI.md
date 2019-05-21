# TensorFlow中的Dataset API

## 概述

使用Dataset的三个步骤：
1. 载入数据：为数据创建一个Dataset实例
2. 创建一个迭代器：使用创建的数据集来构造一个Iterator实例以遍历数据集
3. 使用数据：使用创建的迭代器，我们可以从数据集中获取数据元素，从而输入到模型中去。

## 载入数据

### 从numpy载入
这是最常见的情况，假设我们有一个numpy数组，我们想将它传递给TensorFlow
```
# create a random vector of shape (100,2)
x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
```
我们也可以传递多个numpy数组，最典型的例子是当数据被划分为特征和标签的时候：
```
features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
```

### 从tensors中载入
我们也可以用一些张量初始化数据集
```
# using a tensor
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))
```

### 从placeholder中载入
如果我们想动态地改变Dataset中的数据，使用这种方式是很有用的。
```
x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)
```

### 从generator载入
我们也可以从generator中初始化一个Dataset。当一个数组中元素长度不相同时，使用这种方式处理是很有效的。（例如一个序列）
```
sequence = np.array([[1],[2,3],[3,4]])

def generator():
    for el in sequence:
        yield el
        
dataset = tf.data.Dataset().from_generator(generator, output_types=tf.float32, output_shapes=[tf.float32])
```
在这种情况下，需要指定数据的类型和大小以创建正确的tensor


## 创建一个迭代器

我们已经知道了如何创建数据集，但是如何从中获取数据呢？我们需要使用一个Iterator遍历数据集并重新得到数据真实值。有四种形式的迭代器。

### One shot Iterator
这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次， 遍历完后会报tf.errors.OutOfRangeError，这是最简单的迭代器，下面给出第一个例子：
```
x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)

# create the iterator
iter = dataset.make_one_shot_iterator()
```
接着你需要调用get_next()来获得包含数据的张量
```
...
# create the iterator
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
```
我们可以运行 el 来查看它们的值。
```
with tf.Session() as sess:
    print(sess.run(el)) # output: [ 0.42116176  0.40666069]
```
在Eager模式中，创建Iterator的方式有所不同。是通过tfe.Iterator(dataset)的形式直接创建Iterator并迭代。迭代时可以直接取出值，不需要使用sess.run()：
```
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

for one_element in tfe.Iterator(dataset):
    print(one_element)
```

### 可初始化的迭代器
如果我们想建立一个可以在运行时改变数据源的动态数据集，我们可以用placeholder 创建一个数据集。接着用常见的feed-dict机制初始化这个placeholder。这些工作可以通过使用一个可初始化的迭代器完成。使用上一节的第三个例子
```
# using a placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)
data = np.random.sample((100,2))
iter = dataset.make_initializable_iterator() # create the iterator
el = iter.get_next()
with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iter.initializer, feed_dict={ x: data }) 
    print(sess.run(el)) # output [ 0.52374458  0.71968478]
```
这次，我们调用make_initializable_iterator。接着我们在 sess 中运行 initializer 操作，以传递数据，这种情况下数据是随机的 numpy 数组。

假设我们有了训练集和测试集，如下代码所示
```
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))
```
接着，我们训练该模型，并在测试数据集上对其进行测试，这可以通过训练后对迭代器再次进行初始化来完成。
```
# initializable iterator to switch between dataset
EPOCHS = 10
x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))
iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
with tf.Session() as sess:
#     initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
    for _ in range(EPOCHS):
        sess.run([features, labels])
#     switch to test data
    sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
    print(sess.run([features, labels]))
```

### 可重新初始化的迭代器
这个概念和之前的相似，我们想在数据间动态切换。但是我们是转换数据集而不是把新数据送到相同的数据集。和之前一样，我们需要一个训练集和一个测试集
```
# making fake data using numpy
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
```
接下来创建两个Dataset
```
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
```
现在我们要用到一个小技巧，即创建一个通用的Iterator
```
# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
```
接着创建两个初始化运算
```
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)
```
和之前一样，我们得到下一个元素
```
features, labels = iter.get_next()
```
现在，我们可以直接使用session运行两个初始化运算。把上面这些综合起来我们可以得到：
```
# Reinitializable iterator to switch between Datasets
EPOCHS = 10
# making fake data using numpy
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
features, labels = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)
with tf.Session() as sess:
    sess.run(train_init_op) # switch to train dataset
    for _ in range(EPOCHS):
        sess.run([features, labels])
    sess.run(test_init_op) # switch to val dataset
    print(sess.run([features, labels]))
```

### Feedable迭代器
老实说，我并不认为这种迭代器有用。这种方式是在迭代器之间转换而不是在数据集间转换，比如在来自make_one_shot_iterator()的一个迭代器和来自make_initializable_iterator()的一个迭代器之间进行转换。


## 使用数据

在之前的例子中，我们使用session来打印Dataset中next元素的值
```
...
next_el = iter.get_next()
...
print(sess.run(next_el)) # will output the current element
```
现在为了向模型传递数据，我们只需要传递get_next()产生的张量。

在下面的代码中，我们有一个包含两个numpy数组的Dataset，这里用到了和第一节一样的例子。注意到我们需要将.random.sample封装到另外一个numpy数组中，因此会增加一个维度以用于数据batch。
```
# using two numpy arrays
features, labels = (np.array([np.random.sample((100,2))]), 
                    np.array([np.random.sample((100,1))]))
dataset = 
tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
```
接下来和平时一样，我们创建一个迭代器
```
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
```
建立一个简单的神经网络模型
```
# make a simple model
net = tf.layers.dense(x, 8) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8)
prediction = tf.layers.dense(net, 1)
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)
```
我们直接使用来自iter.get_next()的张量作为神经网络第一层的输入和损失函数的标签。将上面的综合起来可以得到：
```
EPOCHS = 10
BATCH_SIZE = 16
# using two numpy arrays
features, labels = (np.array([np.random.sample((100,2))]), 
                    np.array([np.random.sample((100,1))]))
dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
# make a simple model
net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
```
输出：
```
Iter: 0, Loss: 0.1328 
Iter: 1, Loss: 0.1312 
Iter: 2, Loss: 0.1296 
Iter: 3, Loss: 0.1281 
Iter: 4, Loss: 0.1267 
Iter: 5, Loss: 0.1254 
Iter: 6, Loss: 0.1242 
Iter: 7, Loss: 0.1231 
Iter: 8, Loss: 0.1220 
Iter: 9, Loss: 0.1210
```

## 有用的技巧

### batch
通常情况下，batch是一件麻烦的事情，但是通过Dataset API我们可以使用batch(BATCH_SIZE)方法自动地将数据按照指定的大小batch，默认值是1。在接下来的例子中，我们使用的batch大小为4。
```
# BATCHING
BATCH_SIZE = 4
x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el)) 
```
输出：
```
[[ 0.65686128  0.99373963]
 [ 0.69690451  0.32446826]
 [ 0.57148422  0.68688242]
 [ 0.20335116  0.82473219]]
```


### Repeat
使用.repeat()我们可以指定数据集迭代的次数。如果没有设置参数，则迭代会一直循环，没有结束，因此也不会抛出tf.errors.OutOfRangeError异常。通常来说，一直循环并直接用标准循环控制epoch的次数能取得较好的效果。

### Shuffle
我们可以使用shuffle()方法将Dataset随机洗牌，默认是在数据集中对每一个epoch洗牌，这种处理可以避免过拟合。

我们也可以设置buffer_size参数，下一个元素将从这个固定大小的缓存中按照均匀分布抽取。例子：
```
# BATCHING
BATCH_SIZE = 4
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    print(sess.run(el))
```
首次运行输出：
```
[[4]
 [2]
 [3]
 [1]]
```
第二次运行输出：
```
[[3]
 [1]
 [2]
 [4]]
```
这样数据就被洗牌了。你还可以设置seed参数

### Map
你可以使用map()方法对数据集的每个成员应用自定义的函数。在下面的例子中，我们将每个元素乘以2。
```
# MAP
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.map(lambda x: x*2)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
#     this will run forever
        for _ in range(len(x)):
            print(sess.run(el))
```
输出：
```
[2]
[4]
[6]
[8]
```

## 参考
[如何使用TensorFlow中的Dataset API](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/79342369)
[TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)
