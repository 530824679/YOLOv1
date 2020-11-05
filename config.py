# -*- coding:utf-8 -*-

import os

path_params = {
    'data_path': '/home/chenwei/HDD/datasets/object_detection/VOCdevkit/VOC2012',
    'checkpoints_dir': './checkpoints',
    'logs_dir': './logs',
    'tfrecord_dir': './tfrecord',
    'checkpoints_name': 'model.ckpt',
    'train_tfrecord_name': 'train.tfrecord',
    'test_tfrecord_name': 'test.tfrecord'
}

model_params = {
    'image_size': 448,              # 输入图片的尺寸
    'channels': 3,                  # 输入图片通道数
    'cell_size': 7,                 # 输出特征图的网格大小
    'boxes_per_cell': 2,            # 每个网格负责预测的BBox个数
    'alpha': 0.1,                   # lrelu参数
    'keep_prob': 0.5,               # dropout比例
    'object_scale': 1.0,            # 置信度有目标权重
    'noobject_scale': 0.5,          # 置信度无目标权重
    'class_scale': 2.0,             # 分类损失权重
    'coord_scale': 5.0,             # 定位损失权重
    'num_classes': 20,              # 数据集的类别个数
}

solver_params = {
    'gpu': '0',                     # 使用的gpu索引
    'learning_rate': 0.001,         # 初始学习率
    'batch_size': 8,                # 每批次输入的数据个数
    'max_iter': 10000,              # 训练的最大迭代次数
    'save_step': 100,               # 权重保存间隔
    'log_step': 100,                # 日志保存间隔
    'display_step': 10,             # 显示打印间隔
    'flipped': True,                # 支持数据翻转
    'restore': True                 # 支持restore
}

test_params = {
    'threshold': 0.2,               # 类别置信度分数阈值
    'iou_threshold': 0.4,           # iou阈值，小于0.4被过滤掉
    'max_output_size': 10           # nms选择的边界框最大数量
}

CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

colors = [[156,102,31], [255,127,80], [255,99,71], [255,255,0], [255,153,18],
          [227,207,87], [255,255,255], [202,235,216], [192,192,192], [251,255,242],
          [160,32,240], [218,112,214], [0,255,0], [255,0,0], [25,25,112],
          [3,168,158], [128,138,135], [128,118,105], [160,82,45], [8,46,84]]

Class_to_index = {_class: _index for _index, _class in enumerate(CLASS)}

Colors_to_map = {_class: _color for _class, _color in zip(CLASS, colors)}