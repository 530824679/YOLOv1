import os
import cv2
import numpy as np
import tensorflow as tf
from cfg.config import path_params, model_params
from model import network
from utils.process_utils import *

def predict(test_dir, checkpoints):
    """
    本函数用于对测试
    :param test_dir:待测试的目录
    :param checkpoints:权重文件
    :return:
    """
    input = tf.placeholder(tf.float32, [None, model_params['image_size'], model_params['image_size'], model_params['channels']], name='input')

    # 构建网络
    Model = network.Network(is_train=False)
    logits = Model._build_network(input)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoints)

        file_list = os.listdir(test_dir)
        for filename in file_list:
            file = os.path.join(test_dir, filename)

            image = cv2.imread(file)
            image_width = np.shape(image)[0]
            image_height = np.shape(image)[1]
            image = cv2.resize(image, (model_params['image_size'], model_params['image_size']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = (image / 255.0) * 2.0 - 1.0

            batch_image = np.zeros([1, model_params['image_size'], model_params['image_size'], model_params['channels']])
            batch_image[0, :, :, :] = image

            output = sess.run(logits, feed_dict={input: batch_image})
            result = post_processing(output)

            for i in range(len(result)):
                result[i][1] *= (1.0 * image_width / model_params['image_size'])
                result[i][2] *= (1.0 * image_height / model_params['image_size'])
                result[i][3] *= (1.0 * image_width / model_params['image_size'])
                result[i][4] *= (1.0 * image_height / model_params['image_size'])

            draw_results(file, result)

if __name__ == '__main__':
    test_dir = path_params['test_output_dir']
    checkpoints_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']

    checkpoints_file = os.path.join(checkpoints_dir, checkpoints_name)
    predict(test_dir, checkpoints_file)