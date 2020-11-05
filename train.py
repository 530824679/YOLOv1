# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf

from config import path_params, model_params, solver_params
from model import network
from utils import loss
from data import dataset


def train():
    start_step = 0
    log_step = solver_params['log_step']
    display_step = solver_params['display_step']
    restore = solver_params['restore']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    log_dir = path_params['logs_dir']

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    data = dataset.Dataset()
    image_batch, label_batch = data.load_tfrecord()

    # 定义输入的占位符
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, model_params['image_size'], model_params['image_size'], model_params['channels']], name='inputs')
    labels = tf.placeholder(dtype=tf.float32, shape=[None, model_params['cell_size'], model_params['cell_size'], 5 + model_params['num_classes']], name='labels')

    # 构建网络，预测值shape=[batch_size, cell_size * cell_size * (class_num+5)]
    Model = network.Network(is_train=True)
    logits = Model._build_network(inputs)

    # 预测值和真实值比较，计算loss
    Losses = loss.Loss
    Losses.loss_layer(logits, labels)
    total_loss = tf.losses.get_losses()
    tf.summary.scalar('total_loss', total_loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_setp = tf.train.AdamOptimizer(solver_params['learning_rate']).minimize(total_loss)

    global_step = tf.train.create_global_step()
    saver = tf.train.Saver(var_list=tf.global_variables_initializer(), max_to_keep=1000)

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
    with tf.Session(config=config) as sess:
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            stem = os.path.basename(ckpt.model_checkpoint_path)
            restore_step = int(stem.split('.')[0].split('-')[-1])
            start_step = restore_step
            sess.run(global_step.assign(restore_step))
            saver.restore(sess, ckpt.model_checkpoint_path)

        coordinate = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coordinate, sess=sess)

        for epoch in range(start_step + 1, solver_params['max_iter']):
            start_time = time.time()

            if coordinate.should_stop():
                break
            image, label = sess.run([image_batch, label_batch])



            end_time = time.time()

            if epoch % solver_params['save_step'] == 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)
                print('Save modle into {}....'.format(save_path))
            if epoch % log_step == 0:
                summary = sess.run(summary_op, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=epoch)
            if epoch % display_step == 0:
                per_iter_time = end_time - start_time
                print("step:{:.0f}  total_loss:  {:.5f} {:.2f} s/iter".format(epoch, total_loss, per_iter_time))

        coordinate.join(threads)
        coordinate.request_stop()
        sess.close()
