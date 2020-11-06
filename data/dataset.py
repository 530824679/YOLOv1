import cv2
import os
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET
from cfg.config import path_params, model_params, solver_params, CLASS

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.test_tfrecord_name = path_params['test_tfrecord_name']
        self.image_size = model_params['image_size']
        self.cell_size = model_params['cell_size']
        self.class_num = model_params['num_classes']
        self.class_ind = dict(zip(CLASS, range(self.class_num)))
        self.batch_size = solver_params['batch_size']
        self.flipped = solver_params['flipped']

    def image_read(self, image_num):
        image_path = os.path.join(self.data_path, 'JPEGImages', image_num+'.jpg')
        image = cv2.imread(image_path)

        self.h_ratio = self.image_size / image.shape[0]
        self.w_ratio = self.image_size / image.shape[1]

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2 - 1

        if self.flipped == True:
            return image, image[:, ::-1, :]
        return image

    def make_label(self, image_num):
        label = np.zeros([self.cell_size, self.cell_size, 5 + self.class_num], np.float32)
        label_path = os.path.join(self.data_path, 'Annotations', image_num+'.xml')

        if self.flipped:
            label_flipped = np.zeros([self.cell_size, self.cell_size, 5 + self.class_num], np.float32)

        tree = ET.parse(label_path)
        root = tree.getroot()
        # 得到某个xml_file文件中所有的object
        objects = root.findall('object')
        for object in objects:
            bndbox = object.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text

            x1 = max(min(float(xmin) * self.w_ratio, self.image_size - 1), 0)
            y1 = max(min(float(ymin) * self.h_ratio, self.image_size - 1), 0)
            x2 = max(min(float(xmax) * self.w_ratio, self.image_size - 1), 0)
            y2 = max(min(float(ymax) * self.h_ratio, self.image_size - 1), 0)
            class_index = self.class_ind[object.find('name').text.lower().strip()]

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 -y1

            # 中心点相对于cell×cell网格的索引
            center_x_index = int(center_x / self.image_size * self.cell_size)
            center_y_index = int(center_y / self.image_size * self.cell_size)

            # 对每个object,如果这个cell中有object了,则跳过标记
            if label[center_y_index, center_x_index, 0] == 1:
                continue
            label[center_y_index, center_x_index, 0] = 1
            label[center_y_index, center_x_index, 1:5] = [center_x, center_y, width, height]
            label[center_y_index, center_x_index, 5 + class_index] = 1

            if self.flipped:
                label_flipped[center_y_index, center_x_index, 0] = 1
                label_flipped[center_y_index, center_x_index, 1:5] = [self.image_size - 1 - center_x, center_y, width, height]
                label_flipped[center_y_index, center_x_index, 5 + class_index] = 1

        if self.flipped:
            return label, label_flipped[:, ::-1, :]
        else:
            return label

    # 定义函数转化变量类型,在将样本图片及标定数据写入tfrecord文件之前需要对两者的数据类型进行转换
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # 数组形式的数据,首先转换为string,再转换为二进制形式进行保存
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_tfrecord(self):
        # 获取作为训练验证集的图片序列
        trainval_path = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        if self.flipped:
            tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
            if not os.path.exists(tf_file):
                # 循环写入每一张图像和标签到tfrecord文件
                writer = tf.python_io.TFRecordWriter(tf_file)
                with open(trainval_path, 'r') as read:
                    lines = read.readlines()
                    for line in lines:
                        image_num = line[0:-1]

                        # 获得当前样本数据和标签信息
                        image, image_flipped = self.image_read(image_num=image_num)
                        label, label_flipped = self.make_label(image_num=image_num)

                        # 转换为字符串
                        image_string = image.tostring()
                        image_flipped_string = image_flipped.tostring()

                        # 转换为字符串
                        label_string = label.tostring()
                        label_flipped_string = label_flipped.tostring()

                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image': self._bytes_feature(image_string),
                                'label': self._bytes_feature(label_string)}))
                        writer.write(example.SerializeToString())
                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image': self._bytes_feature(image_flipped_string),
                                'label': self._bytes_feature(label_flipped_string)}))
                        writer.write(example.SerializeToString())
                writer.close()
                print('Finish trainval.tfrecord Done')
        else:
            tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
            if not os.path.exists(tf_file):
                # 循环写入每一张图像和标签到tfrecord文件
                writer = tf.python_io.TFRecordWriter(tf_file)
                with open(trainval_path, 'r') as read:
                    lines = read.readlines()
                    for line in lines:
                        image_num = line[0:-1]
                        image = self.image_read(image_num)
                        label = self.make_label(image_num)

                        image_string = image.tostring()
                        label_string = label.tostring()

                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
                            }))
                        writer.write(example.SerializeToString())
                writer.close()
                print('Finish trainval.tfrecord Done')


    def load_tfrecord(self):

        tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)

        filename_queue = tf.train.string_input_producer([tf_file])

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        img_features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(img_features['image'], tf.float32)

        label = tf.decode_raw(img_features['label'], tf.float32)

        image = tf.reshape(image, [self.image_size, self.image_size, 3])

        label = tf.reshape(label, [self.cell_size, self.cell_size, 5 + self.class_num])

        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.batch_size, capacity=2000, min_after_dequeue=100)

        return image_batch, label_batch
