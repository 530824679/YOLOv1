# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :preprocess data
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET
from cfg.config import path_params, model_params, solver_params, CLASSES

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.test_tfrecord_name = path_params['test_tfrecord_name']
        self.image_size = model_params['image_size']
        self.cell_size = model_params['cell_size']
        self.class_num = model_params['num_classes']
        self.class_ind = dict(zip(CLASSES, range(self.class_num)))
        self.batch_size = solver_params['batch_size']
        self.flipped = solver_params['flipped']

    def load_image(self, image_num):
        '''
        依据image_num对相应的样本图片进行加载，同时执行resize操作，并对图像进行归一化操作
        :param image_num: 图片编号
        :return: 归一化后的图片数据
        '''
        image_path = os.path.join(self.data_path, 'JPEGImages', image_num+'.jpg')
        image = cv2.imread(image_path)

        self.h_ratio = 1.0 * self.image_size / image.shape[0]
        self.w_ratio = 1.0 * self.image_size / image.shape[1]

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2 - 1

        if self.flipped == True:
            return image, image[:, ::-1, :]
        return image

    def load_annotation(self, image_num):
        '''
        对单个xml文件进行解析,输出为该图片样本对应的label矩阵，label为三维矩阵[cell_size, cell_size, 5 + 20],
        confidence取值方法为:当前单元格包含目标则为１，不包含目标为０；(x, y, w, h)为box的形状信息，中心坐标，宽，高，均以像素坐标的形式给出
        :param image_num: 图片编号
        :return: 归一化后的图片数据
        '''
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

            # 将原始样本的标定转换为resize后的图片的标定,按照等比例转换的方式,从0开始索引
            x1 = max(min(float(xmin) * self.w_ratio, self.image_size - 1), 0)
            y1 = max(min(float(ymin) * self.h_ratio, self.image_size - 1), 0)
            x2 = max(min(float(xmax) * self.w_ratio, self.image_size - 1), 0)
            y2 = max(min(float(ymax) * self.h_ratio, self.image_size - 1), 0)

            # 将类别由字符串转换为对应的int数
            class_index = self.class_ind[object.find('name').text.lower().strip()]

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 -y1

            # 计算当前目标属于第几个cell，从0开始索引
            center_x_index = int(center_x / self.image_size * self.cell_size)
            center_y_index = int(center_y / self.image_size * self.cell_size)

            # 对每个object,如果这个cell中有object了,则跳过标记
            if label[center_y_index, center_x_index, 0] == 1:
                continue

            # 这里的x, y坐标是交换的，原因在于numpy的索引和图像的索引是颠倒的，在图像中０维索引列，１维索引行
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
