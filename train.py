# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cfg.config import path_params, model_params, solver_params
from model import network
from utils import loss_utils
from data import dataset, tfrecord


def train():
    start_step = 0
    log_step = solver_params['log_step']
    display_step = solver_params['display_step']
    restore = solver_params['restore']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    tfrecord_name = path_params['train_tfrecord_name']
    log_dir = path_params['logs_dir']

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, tfrecord_name)
    image_batch, label_batch = data.parse_batch_examples(train_tfrecord)

    # 定义输入的占位符
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, model_params['image_size'], model_params['image_size'], model_params['channels']], name='inputs')
    outputs = tf.placeholder(dtype=tf.float32, shape=[None, model_params['cell_size'], model_params['cell_size'], 5 + model_params['num_classes']], name='outputs')

    # 构建网络
    Model = network.Network(is_train=True)
    logits = Model._build_network(inputs)

    # 计算损失函数
    Losses = loss_utils.Loss(logits, outputs, 'loss')
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('total_loss', total_loss)

    global_step = tf.train.create_global_step()

    # 设置优化器
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(solver_params['learning_rate'])
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)
        #train_op = optimizer.minimize(total_loss, global_step=global_step)

    # 模型保存
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=1000)

    # 配置tensorboard
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    with tf.Session(config=config) as sess:
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_step = int(stem.split('.')[0].split('-')[-1])
                start_step = restore_step
                sess.run(global_step.assign(restore_step))
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            else:
                print("Failed to find a checkpoint")

        coordinate = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coordinate, sess=sess)
        summary_writer.add_graph(sess.graph)

        for epoch in range(start_step + 1, solver_params['max_iter']):
            start_time = time.time()

            if coordinate.should_stop():
                break
            image, label = sess.run([image_batch, label_batch])
            feed_dict = {inputs: image, outputs: label}
            _, loss, current_global_step = sess.run([train_op, total_loss, global_step], feed_dict=feed_dict)

            end_time = time.time()

            if epoch % solver_params['save_step'] == 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)
                print('Save modle into {}....'.format(save_path))

            if epoch % log_step == 0:
                summary = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=epoch)

            if epoch % display_step == 0:
                per_iter_time = end_time - start_time
                print("step:{:.0f}  total_loss:  {:.5f} {:.2f} s/iter".format(epoch, loss, per_iter_time))

        coordinate.request_stop()
        coordinate.join(threads)
        sess.close()

if __name__ == '__main__':
    train()