# tensorflow-MonitoredTrainingSession解读

### 一、 训练为什么要管理
搭建一个简单的分布式训练是不需要管理的，只需要定义好ClusterSpec，给每个节点分配Server，建好图，就可以开始迭代了

随着问题和模型的复杂化，我们也许会有监控训练的需求，如记录日志、训练可视化、checkpoint、early-stop、训练效率调优等，tensorflow提供了大量的工具支持，但这就加重了代码的复杂度。所以tensorflow封装了MonitoredTrainingSession，将各种监控训练的组件外挂到一个类

###  二、 MonitoredTrainingSession参数

```
    tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        scaffold=None,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=USE_DEFAULT,
        save_summaries_steps=USE_DEFAULT,
        save_summaries_secs=USE_DEFAULT,
        config=None,
        stop_grace_period_secs=120,
        log_step_count_steps=100,
        max_wait_secs=7200,
        save_checkpoint_steps=USE_DEFAULT,
        summary_dir=None
    )
```

args:

- master: server.target
- is_chief: 是否为chief（一般把task_index=0定为chief）。chief节点会负责初始化和模型restore，其他节点只需等待chief初始化完成
- checkpoint_dir: checkpoint文件路径
- scaffold：用于完成图表
- hooks：最重要的参数。它是一个SessionRunHook对象的列表，包含了所有希望外挂的组件，如**CheckpointSaverHook、FeedFnHook、LoggerTensorHook、NanTensorHook、ProfileHook、StopAtStepHook等，也可以自定义Hook，只要继承SessionRunHook类就行**。下面会详细介绍几个重要Hook
- chief_only_hooks：只有chief节点才会生效的hook
- save_checkpoint_secs：保存checkpoint的频率
- save_summaries_steps：按步数保存summary的频率 ；save_summaries_secs是按时间
- config：session配置，是ConfigProtoproto格式

实例化后就得到一个MonitoredSession对象，可以当作普通session使用


### 三、 Hook的使用

Hook顾名思义，是一个“外挂”的组件，用于执行训练中的各种功能。

Hook的基类是tf.train.SessionRunHook，需要实现下面几个方法：

1. 在session被创建后调用： `after_create_session(session, coord)`
2. 在每次session.run后被调用： `after_run(run_context,run_values)`
3. 每次run前调用: `before_run(run_context)`
4. 调用后，图就不能再修改: `begin()`
5. 结束session: `end(session)`

几个常用的内置的Hook如下：

1. tf.train.StopAtStepHook：在一定步数停止。
```
    __init__(
        num_steps=None,
        last_step=None
    )
```
两个参数只能设一个，num_steps是执行步数，last_step是终止步数。

2. tf.train.CheckpointSaverHook：checkpoint保存
```
    __init__(
        checkpoint_dir,
        save_secs=None,
        save_steps=None,
        saver=None,
        checkpoint_basename='model.ckpt',
        scaffold=None,
        listeners=None
    )
```
参数设置了checkpoint的路径、保存频率、saver等

3. tf.train.FeedFnHook：创建feed_dict
```
    __init__(feed_fn)
```
指定生成feed的函数

4. tf.train.FinalOpsHook：在session结束时的评估操作
```
    __init__(
        final_ops,
        final_ops_feed_dict=None
    )
```
在训练结束时，final_ops_feed_dict 喂给final_ops这个tensor，得到final_ops_values。一般用来做测试集的评估

5. tf.train.NanTensorHook：监控loss是否为NAN
```
    __init__(
        loss_tensor,
        fail_on_nan_loss=True
    )
```
调试和终结训练用。如果可以正常训练，建议不用这个Hook，对效率影响比较大

6. tf.train.SummarySaverHook：记录summary，训练可视化
```
    __init__(
        save_steps=None,
        save_secs=None,
        output_dir=None,
        summary_writer=None,
        scaffold=None,
        summary_op=None
    )
```
给定summary_op，定期输出。

7. 自定义Hook。可以自己实现Hook，只要继承SessionRunHook，实现几个方法即可。给一个cifar10中定义LoggerHook的例子：
```
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time#duration持续的时间
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
```
该Hook定制了各种记录日志的方法


### 四、总结
MonitoredTrainingSession和Hook的结合使得可以自由组装训练过程，配合分布式训练和tensorboard的使用，可以提高调试效率。
