# /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import tarfile
from six.moves import urllib

import tensorflow as tf
import cifar10_input


# 全局常量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# 模型常量
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0        # 多少周期岁衰减一次
LEARNING_RATE_DECAY_FACTOR = 0.1    # 权重衰减因子
INITIAL_LEARNING_RATE = 0.1         # 最初学习率

# Multi-GPU运算时, 前缀名
TOWER_NAME = "tower"

# 数据集地址
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def inference(images):
    """
    网络模型, 前向传播
    Args:
        images: 4-D (batch_size, h, w, c) size tensor, tf.float32
    Return:
        labels: 1-D (batch_size, ) size
    """
    # 第一层, 卷积: conv1
    with tf.variable_scope('conv1') as scope:
        # 获取kernel, 采用tf.get_variable()方法: 4-D [filter_height, filter_width, in_channels, out_channels]
        kernel = _variable_with_weight_decay(name='weight', shape=(5,5, 3, 64), stddev=5e-2, wd=0.0)        # TODO: 定义kernnel
        # 计算卷积
        conv = tf.nn.conv2d(input=images, filter=kernel, strides=(1,1,1,1), padding='SAME')
        # 定义bias
        biases = _variable_on_cpu(name='biases', shape=(64,), )                                         # TODO: 定义bias
        # 激活前
        pre_activation = tf.nn.bias_add(conv, biases)
        # relu激活并输出
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        print('scope.name:', scope.name)

        # 为conv层输出, 添加summary
        _activation_summary(conv1)

    # 池化层: pool1
    pool1 = tf.nn.max_pool(conv1, ksize=(1,3,3,1), strides=(1,2,3,1), padding='SAME', name='pool1')

    # 局部响应归一化: tf.nn.local_response_normalization, 貌似没什么卵用...
    norm1 = tf.nn.lrn(input=pool1, depth_radius=4, biases=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # 第二层, 卷积: conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(name='weight', shape=(5,5,64), stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(input=norm1, filter=kernel, strides=(1,1,1,1), padding='SAME')
        biases = _variable_on_cpu(name='biases', shape=(64,), initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation)
        _activation_summary(conv2)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME', name='pool2')

    # 第3层, 全连接: local3
    with tf.variable_scope('local3') as scope:
        with tf.Session() as sess:
            batch_size = sess.run(images).shape[0]

        # Flatten: 每个样本flatten成一个行向量
        reshape = tf.reshape(pool2, shape=(batch_size, -1))
        # 获取样本向量维度
        dim = reshape.get_shape()[1].value

        # 定义权重weight
        weights = _variable_with_weight_decay(name='weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        # 定义偏差biases
        biases = _variable_on_cpu(name='biases', shape=(384,), initializer=tf.constant_initializer(0.1))

        # 计算local3全连接层输出
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # 为该层输出添加summary
        _activation_summary(local3)

    # 第4层, 全连接: local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(name='weights', shape=(384, 192), stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(name='biases', shape=(192,), initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # 第5层, 线性输出: linear_layer, 不做softmax, loss以内置
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(name='weights', shape=(192, NUM_CLASSES), stddev=1/192, wd=0.0)
        biases = _variable_on_cpu(name='biases', shape=(NUM_CLASSES,), initializer=tf.constant_initializer(0.0))

        # 只计算线性输出, 无激活, 无softmax
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear
   

def loss(labels_pred, labels_real):
    """
    计算loss值, 包括交叉熵损失, 及L2正则项
    Args:
        labels_pred: 预测值, (batch_size, ) tensor  # 查看shape是否为(batch_size,)
        labels_real: 真实值, (batch_size, ) tensor
    Return:
        total_loss: 包括交叉熵损失, 及 L2 正则化
    """
    # 转换label标签类型
    labels_real = tf.cast(labels_real, tf.int32)

    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_real,
                                                                 logits=labels_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 将交叉熵添加到Losses集合
    tf.add_to_collection(name='losses', value=cross_entropy_mean)

    # total loss: losses集合内所有loss相加, 包括交叉熵和L2
    total_loss = tf.add_n(tf.get_collection(name='losses'), name='total_loss')

    return total_loss


def train_operator(loss, global_step, batch_size):
    """
    定义一个优化器, 对所有可优化参数进行更新, 采用滑动平均法
    Args:
        loss: total loss consists of loss and weight decay, eg. L2 + cross_entropy
        global_step: 迭代步数
    Return:
        train_op: 优化器, op for training
    """

    # 定义learning_rate: 指数衰减, 每decay_steps次, lr * decay_rate
    # 公式为: decay_lr = lr * decay_rate ^ (global_step / decay_steps)
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                    decay_steps=decay_steps, decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # 添加summary监视器
    tf.summary.scalar(name='learning_rate', tensor=lr)

    # 对loss做平滑处理, 及添加监视器
    loss_averages_op = _add_loss_summaries(loss)        # TODO: loss函数平滑处理

    # 优化, 参数更新: 计算梯度, 及应用梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(loss)         # 为何是loss, 而非loss_averages
    apply_gradient_op = opt.apply_gradients(grads_and_vars=grads, global_step=global_step)

    # 为trainable_variable添加sunmmary
    for var in tf.trainable_variables:
        tf.summary.histogram(name=var.op.name, values=var)

    # 为variable的gradient添加summay
    for grad, var in grads:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # 定义"变量的滑动平均"作滑动平均, 衰减因子: decay = min(decay, (1+gboal_steps) / (10+global_steps)_
    # 公式: shadow_variable = decay * shadow_variable + (1-decay) * variable
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_aveages_op = variable_averages.apply(var_list=tf.trainable_variables())

    # 定义依赖关系, 及定义一个空op, 命名为train_op
    with tf.control_dependencies([apply_gradient_op, variable_aveages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(loss):
    """
    对loss做滑动平均, 及添加summary
    Args:
        loss: total_loss
    Return:
        loss_averages_op:  op for generating moving averages of losses
    """
    # 定义滑动平均
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9, name='avg')
    # 取出losses集合的所有loss
    losses = tf.get_collection(key='losses')
    # 对所有loss做滑动平均
    loss_averages_op = loss_averages.apply(losses + [loss])

    # 对losses集合的每个loss, 添加summary
    for l in losses + [loss]:
        tf.summary.scalar(l.op.name + 'raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _activation_summary(x):
    pass





def _variable_with_weight_decay(name, shape, stddev, wd):
    pass


def _variable_on_cpu(name, shape, initializer):
    pass