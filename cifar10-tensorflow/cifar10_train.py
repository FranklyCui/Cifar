# /usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import time
import os

import tensorflow as tf
import numpy as np

import cifar10_model
import cifar10_input
from six.moves import xrange


# 定义全局变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(name='dataset_dir', default='~/Tensorflow_learning/Cifar-10/cifar10-tensorflow/dataset',
                          help='Directory where dataset stored')
tf.app.flags.DEFINE_string(name='model_dir', default='~/Tensorflow_learning/Cifar-10/cifar10-tensorflow/model',
                           help="Directory where to write event logs and checkpoint.")

tf.app.flags.DEFINE_integer(name='max_steps', default=1000000, help="Number of batches to run.")
tf.app.flags.DEFINE_integer(name='batch_size', default=128, help='Number of images process in a batch')

tf.app.flags.DEFINE_boolean(name='log_device_placement', default=False, help="Whether to log device placement.")



def train():
    """
    Train cifar-10 for a number of steps.
    """
    with tf.Graph().as_default():
        # 创建全局步数
        global_step = tf.train.get_or_create_global_step()

        # 获取训练数据
        images, labels = cifar10_input.distorted_input(data_dir=FLAGS.dataset_dir, batch_size=FLAGS.batch_size)

        # 获取预测值
        labels_pred = cifar10_model.inference(images)        # TODO: cifar10改为cifar10_model

        # 计算损失值
        loss = cifar10_model.loss(labels_pred, labels)

        # 定义训练器
        train_op = cifar10_model.train_operator(loss, global_step, batch_size=FLAGS.batch_size)

        # 定义Saver
        saver = tf.train.Saver(tf.all_variables())

        # 定义summary_op
        summary_op = tf.summary.merge_all()

        # 初始化全局变量
        init = tf.global_variables_initializer()

        # 开启session
        sess = tf.Session()         # TODO: 配置运算设备, 默认是cpu还是GPU?
        sess.run(init)

        # 启动队列
        tf.train.start_queue_runners(sess=sess)

        # 定义summary存储器
        summary_writer = tf.summary.FileWriter(logdir=FLAGS.model_dir, graph=sess.graph_def)

        # 迭代:
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run((train_op, loss))          # Todo: train, loss需在一个sess.run()内运行
            duration = time.time() - start_time

            # 断定Loss值未发散
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # 每10个step显示: step, loss, time等运行数据
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                print(('%s: step %d, loss=%.2f (%.1f examples / sec; %.3f' ' sec / batch)'
                       % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch)))

            # 每100个step, 运行数据保存在summary内
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary=summary_str, global_step=step)

            # 每1000个step保存一次模型
            if step % 1000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'model', 'model.ckpt')
                saver.save(sess=sess, save_path=checkpoint_path, global_step=global_step)



def main():
    """  主函数: 程序入口  """
    # 下载数据、解压数据
    cifar10_input.maybe_download_and_extract(data_dir=dir)

    # 开始训练
    train()



if __name__ == '__main__':

    # 处理flag解析, 执行main()函数
    tf.app.run()
