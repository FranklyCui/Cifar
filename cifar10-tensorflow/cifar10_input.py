# /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from six.moves import xrange
import tarfile
# import urllib         # python3的urllib包有变
from six.moves import urllib
import sys
import matplotlib.pyplot as plt

# resize后的图像尺寸
IMAGE_SIZE = 24

# 数据集参数
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def download_data(data_dir):
    """  如cifar-10数据集不存在, 则下载数据集并解压缩  """
    maybe_download_and_extract(data_dir)


def maybe_download_and_extract(data_dir, DATA_URL=None):
    """Download and extract the tarball from Alex's website."""
    # 下载链接
    if not DATA_URL:
        DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    # 目标路径
    dest_directory = data_dir
    # 若路径不存在, 则创建
    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    # 截取文件名
    filename = DATA_URL.split('/')[-1]
    print('filename: ', filename)
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def check_files(filenames):
    """  查看Cifar-10数据集是否存在且完整  """
    for file in filenames:
        if not os.path.exists(file):
            print('Not find %s file.' % file)
            raise ValueError('Failed to find file: ' + file)
    return True


def get_record(filenames_queue):
    """
    采用FixedLengthRecoardReader读取器, 从队列中读取'键值对'
    Args:
        filename_queue: 文件名队列
    Returns:
        an object.
    """
    print('get record starting...')
    # 创建结构对象, 以便于保存数据, 并实例化
    class Record(object):
        pass
    record = Record()

    # 定义图像参数
    record.height = 32
    record.width = 32
    record.depth = 3
    label_btytes = 1    # 2 for CIFAR-100, label的字节数
    image_btypes = record.height * record.width * record.depth      # 图片位数
    record_btypes = label_btytes + image_btypes     # Record(含img和label): 位数

    # 构造FixeLengthRecord读取器, 读取文件队列, 返回: 键值对
    reader = tf.FixedLengthRecordReader(record_bytes=record_btypes)
    record.key, value = reader.read(filenames_queue)         # key是什么???
    # 将record字符串数据类型转为tf.unint8数据类型
    record_btypes = tf.decode_raw(value, tf.uint8)

    # 从record数据, 切出label, 转为tf.int32格式
    label = tf.slice(record_btypes, begin=[0], size=[label_btytes])
    record.label = tf.cast(label, tf.int32)

    # 从record数据, 切出imag, 并从(channel, height, width) 转为 (height, width, channel), 转为tf.float32格式
    imag = tf.slice(record_btypes, begin=[label_btytes], size=[image_btypes])
    imag = tf.reshape(imag, shape=(record.depth, record.height, record.width))
    imag = tf.transpose(imag, perm=(1, 2, 0))
    record.imag = tf.cast(imag, tf.float32)

    return record


def get_image(data_dir):
    """ 输入路径, 取出record  """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    print('filenames: ', filenames)
    if not check_files(filenames):
        exit()
    filenames_queue = tf.train.string_input_producer(filenames)
    record = get_record(filenames_queue)
    return record


def distorted_input(data_dir, batch_size):
    """
    为训练, 构造扰动的输入
    Args:
        data_dir: 数据集路径
        batch_size: 每批数据量
    Return:
        images: 4-D tensor of (b, h, w, 3) size
        labels: 1-D tensor of (b, ) size
    """
    # 读取数据, 返回Record()对象
    read_input = get_image(data_dir=data_dir)
    reshape_image = read_input.imag

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 图像增强: 随机裁剪/左右翻转/随机亮度/随机对比度/归一化
    distorted_image = tf.random_crop(value=reshape_image, size=(height, width, 3))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)       # Z-Score归一化: (img-mean)/stddev, 结果: 均值为0, 标准差为1的正态分布
    print('float_image.shape: ', float_image.shape)
    # 设定tensor的shape
    float_image.set_shape(shape=(height, width, 3))
    print('label.shape: ', read_input.label.shape)
    read_input.label.set_shape(shape=(1,))

    # 确保shuffle具有好的效果, 需保证队列中的剩余样本数量大于一定值
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print('Filling queue with %d CIFAR images before starting to train.'
          'This will take a few minutes.' % min_queue_examples)

    # 创建一个batch, 通过构建一个样本队列
    batch = _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)
    print('float_image.shape: ', float_image.shape)
    print('batch.shape: ', batch.shape)

    return batch


def inputs(eval_data, data_dir, batch_size):
    """
    构造输入
    Args:
        eval_data: bool类型, 采用train还是eval数据集
        data_dir: 数据集路径
        batch_size: 每批数据量
    Return:
        images: 4-D tensor of (batch_size, h, w, c) size
        labels: 1-D tensor of (batch_size) size
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # 判断文件是否存在
    check_files(filenames)

    # 创建文件名队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 利用读取器, 从文件名队列读取数据
    read_input = get_record(filenames)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 图样处理: 为evaluation: 中心裁剪
    resize_image = tf.image.resize_image_with_crop_or_pad(read_input, target_height=height, target_width=width)
    # 归一化: Z-Score归一化, 转为均值为0, 方差为1的正态分布
    float_image = tf.image.per_image_standardization(resize_image)

    # 设定图像尺寸
    float_image.set_shape((height, width, 3))
    read_input.label.set_shape((1,))

    # 保证良好的shuffle效果
    min_fraction_of_examples_in_queue = 0.4
    min_queue_example = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # 创建batch
    batch = _generate_image_and_label_batch(float_image, read_input.label, )
    return batch



def _generate_image_and_label_batch(image, label, min_queue_examples, batchs_size, shuffle=True):
    """
        生成batch数据
        Args:
            image: 3-D Tensor of (h, w, 3), type: tf.float32
            label: 1-D Tensor, type: tf.int32
            min_queue_examples: 队列中需保留的最小样本数
            batchs_size: ..
        Return:
            images: 4-D Tensor of (batch_size, h, w, 3) size
            labels: 1-D Tensro of (batch_size) size
    """
    # 创建样本队列
    num_preprocess_threads = 4

    # .batch()的输入, 究竟是(num_examples, h, w, c), 还是(h, w, c)???
    print('image.shape: ', image.shape)
    print('label.shape: ', label.shape)

    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batchs_size, num_threads=num_preprocess_threads,
                                  capacity=min_queue_examples + 3 * batchs_size, min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(tensors=(image, label), batchs_size=batchs_size, num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples + 3 * batchs_size)

    images = tf.cast(images, tf.float32)
    label_batch = tf.cast(label_batch, tf.float32)

    print('*' * 50)
    print('images.shape: ', images.shape)
    print('label_batch.shape: ', label_batch.shape)

    labels = tf.reshape(label_batch, (batchs_size))
    print('labels.shape: ', labels.shape)
    # 可视化
    tf.summary.image(name='images', tensor=images)
    return images, labels


if __name__=='__main__':

    # 下载并解压数据
    data_dir = '/home/frank/Tensorflow_learning/Cifar-10/cifar10-tensorflow/data'
    download_data(data_dir)

    # 图片保存路径
    image_save_dir = './Cifar10_images'
    if os.path.exists(image_save_dir):
        os.mkdir(image_save_dir)

    # 获取图片
    Img = get_image(data_dir)

    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # *** 启动文件名队列(系统会自动启动内存队列) **
        threads = tf.train.start_queue_runners(sess=sess)

        for num in range(200):
            img = sess.run(Img.imag)
            label = sess.run(Img.label)
            key = sess.run(Img.key)
            if num < 3:
                print('*' * 50)
                print('img:', img)
                print('label: ', label)
                print('key: ', key)

            plt.savefig(data_dir + '%d.jpg' % num)