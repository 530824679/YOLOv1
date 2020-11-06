import tensorflow as tf
from cfg.config import model_params, solver_params

class Loss(object):
    def __init__(self):
        self.batch_size = solver_params['batch_size']
        self.image_size = model_params['image_size']
        self.cell_size = model_params['cell_size']
        self.num_class = model_params['num_classes']
        self.boxes_per_cell = model_params['boxes_per_cell']
        self.boundary1 = model_params['cell_size'] * model_params['cell_size'] * model_params['num_classes']
        self.boundary2 = self.boundary1 + model_params['cell_size'] * model_params['cell_size'] * model_params['boxes_per_cell']
        self.class_scale = model_params['class_scale']
        self.object_scale = model_params['object_scale']
        self.noobject_scale = model_params['noobject_scale']
        self.coord_scale = model_params['coord_scale']

    def loss_layer(self, predicts, labels, scope='loss'):
        """
        :param predicts:网络的输出 [batch, cell_size * cell_size * (5 * boxes_per_cell + class_num)]
        :param labels:标签信息 [batch, cell_size, cell_size, 5 + class_num]
        :param scope:命名loss
        """
        # 预测坐标：x, y中心点基于cell, sqrt(w),sqrt(h)基于全图0-1范围
        with tf.name_scope('Predicts Tensor'):
            # 类别预测 predicts reshape ——> [batch_size, 7, 7, 20]
            predicts_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            # 置信度预测 predicts reshape ——> [batch_size, 7, 7, 2]
            predicts_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            # 坐标预测 predicts reshape ——> [batch_size, 7, 7, 2, 4]
            predicts_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        # 标签坐标： x, y, w, h 基于全图0-1范围
        with tf.name_scope('Labels Tensor'):
            # labels reshape ——> [batch_size, 7, 7, 1] 哪个网格负责检测目标就标记为1
            labels_response = tf.reshape(labels[..., 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            # 坐标标签 labels reshape ——> [batch_size, 7, 7, 2, 4] 网格内负责检测的外接框位置以图像大小为基准(x, y, width, height)
            labels_boxes = tf.reshape(labels[..., 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            labels_boxes = tf.tile(labels_boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            # 类别标签 labels reshape ——> [batch, 7, 7, 20]
            labels_classes = labels[..., 5:]


        '''
        # 将网络所预测的bbox相对于cell的偏移量转换为bbox的中心坐标在图像中的比例
        offset = np.transpose(np.reshape(np.array([np.arange(para.cell_size)] * para.cell_size * para.box_per_cell),
                                         (para.box_per_cell, para.cell_size, para.cell_size)), (1, 2, 0))
        # 转换为四维矩阵
        offset = tf.reshape(tf.constant(offset, dtype=tf.float32), [1, para.cell_size, para.cell_size, para.box_per_cell])
        # 将第０维复制batch_size次
        offset = tf.tile(offset, [para.batch_size, 1, 1, 1])
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        '''

        with tf.variable_scope(scope):
            # 类别损失
            class_loss = self.class_loss(predicts_classes, labels_classes, labels_response)
            # 基于cell的x, y 基于全图的sqrt(w), sqrt(h)——>基于全图的x, y, w, h
            global_predict_boxes = self.predicts_to_labels_coord(predicts_boxes)
            # 计算iou [batch , 7, 7, 2]
            iou = self.calc_iou(global_predict_boxes, labels_boxes)
            # 计算有目标和无目标掩码
            object_mask, noobject_mask = self.calc_mask(iou, labels_response)
            # 置信度损失
            object_loss, noobject_loss = self.confidence_loss(predicts_scales, iou, object_mask, noobject_mask)
            # 坐标损失
            boxes_loss = self.coord_loss(predicts_boxes, labels_boxes, object_mask)

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(boxes_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('boxes_loss', boxes_loss)

            tf.summary.histogram('iou', iou)

    def class_loss(self, predicts_class, labels_class, labels_response):
        """
        计算分类损失
        :param predicts_class: 预测类别[batch, 7, 7, 20]
        :param labels_class: 标签类别[batch, 7, 7, 20]
        :param labels_response: cell中是否有目标[batch, 7, 7, 1]
        :return:
        """
        with tf.name_scope('class_loss'):
            class_delta = labels_response * (predicts_class - labels_class)
            class_loss = self.class_scale * tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss')
        return class_loss

    def confidence_loss(self, predicts_scale, iou, object_mask, noobject_mask):
        '''
        计算置信度损失
        :param predicts_scale: 预测置信度 [batch, 7, 7, 2]
        :param iou: iou结果 [batch, 7, 7, 2]
        :param object_mask: 目标掩码 [batch, 7, 7, 2], 有目标位置为1,其余0
        :param noobject_mask: 无目标掩码 [batch, 7, 7, 2], 无目标位置为1,其余0
        :return:
        '''
        with tf.name_scope('confidence loss'):
            with tf.name_scope('object confidence loss'):
                object_confidence_delta = object_mask * (predicts_scale - iou)
                object_confidence_loss = self.object_scale * tf.reduce_mean(tf.reduce_sum(tf.square(object_confidence_delta), axis=[1, 2, 3]))

            with tf.name_scope('no object confidence loss'):
                noobject_confidence_delta = noobject_mask * (predicts_scale - 0)
                noobject_confidence_loss = self.noobject_scale * tf.reduce_mean(tf.reduce_sum(tf.square(noobject_confidence_delta), axis=[1, 2, 3]))
        return object_confidence_loss, noobject_confidence_loss

    def coord_loss(self, predicts_boxes, labels_boxes, object_mask):
        '''
        计算定位损失
        :param predicts_boxes: 预测置位置 基于cell的x, y以及全图 sqrt(w), sqrt(h) [batch, 7, 7, 2, 4]
        :param labels_boxes: 标签位置 基于全图的x, y, w, h [batch, 7, 7, 2, 4]
        :param object_mask: 有目标的掩码 [batch, 7, 7, 2]
        :return:
        '''
        with tf.name_scope('coord_loss'):
            coord_mask = tf.expand_dims(object_mask, axis=-1)
            cell_labals_boxes = self.labels_to_predicts_coord(labels_boxes)
            coord_delta = coord_mask * (predicts_boxes - cell_labals_boxes)
            boxes_loss = self.coord_scale * tf.reduce_mean(tf.reduce_sum(tf.square(coord_delta), axis=[1, 2, 3, 4]))

            tf.summary.histogram('boxes_delta_x', coord_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', coord_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', coord_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', coord_delta[..., 3])
        return boxes_loss


    def predicts_to_labels_coord(self, predicts_boxes):
        # 边界框的中心坐标xy——相对于每个cell左上点的偏移量
        offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0), multiples=[7, 1])
        offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]), multiples=[1, 1, 1, 2, 1])
        offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))

        x = (predicts_boxes[..., 0] + offset_axis_2) / self.cell_size
        y = (predicts_boxes[..., 1] + offset_axis_1) / self.cell_size
        w = tf.square(predicts_boxes[..., 2])
        h = tf.square(predicts_boxes[..., 3])

        global_predicts_boxes = tf.stack([x, y, w, h], axis=-1)
        return global_predicts_boxes

    def labels_to_predicts_coord(self, labels_boxes):
        # 得到x, y相对于该cell左上角的偏移值， 宽度和高度是相对于整张图片的比例
        offset_axis_2 = tf.tile(tf.expand_dims(tf.range(7), axis=0), multiples=[7, 1])
        offset_axis_2 = tf.tile(tf.reshape(offset_axis_2, shape=[1, 7, 7, 1, 1]), multiples=[1, 1, 1, 2, 1])
        offset_axis_1 = tf.transpose(offset_axis_2, (0, 2, 1, 3, 4))

        x = labels_boxes[..., 0] * self.cell_size - offset_axis_2
        y = labels_boxes[..., 1] * self.cell_size - offset_axis_1
        sqrt_w = tf.sqrt(labels_boxes[..., 2])
        sqrt_h = tf.sqrt(labels_boxes[..., 3])

        cell_labals_boxes = tf.stack([x, y, sqrt_w, sqrt_h], axis=-1)
        return cell_labals_boxes

    def calc_iou(self, boxes_1, boxes_2, scope='iou'):
        '''
        计算BBoxes和label的iou
        :param boxes_1: 预测的Boxes [batch, cell, cell, boxes_per_cell, 4] / [x, y, w, h]
        :param boxes_2: 标签的Boxes [batch, cell, cell, boxes_per_cell, 4] / [x, y, w, h]
        :param scope: 命名空间iou
        :return:
        '''
        with tf.name_scope(scope):
            # transform [center_x, center_y, w, h]——>[x1, y1, x2, y2]
            boxes1 = tf.stack([boxes_1[..., 0] - boxes_1[..., 2] / 2.0,
                               boxes_1[..., 1] - boxes_1[..., 3] / 2.0,
                               boxes_1[..., 0] + boxes_1[..., 2] / 2.0,
                               boxes_1[..., 1] + boxes_1[..., 3] / 2.0], axis=-1)
            boxes2 = tf.stack([boxes_2[..., 0] - boxes_2[..., 2] / 2.0,
                               boxes_2[..., 1] - boxes_2[..., 3] / 2.0,
                               boxes_2[..., 0] + boxes_2[..., 2] / 2.0,
                               boxes_2[..., 1] + boxes_2[..., 3] / 2.0], axis=-1)

            lu = tf.maximum(boxes1[..., :2], boxes2[..., :2])
            rd = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

            # [batch, 7, 7, 2] 2个bbox跟label的iou
            intersection_wh = tf.maximum(0.0, rd - lu)
            intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_area = tf.maximum(square1 + square2 - intersection_area, 1e-10)
            return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    def calc_mask(self, iou, response):
        '''
        计算目标/非目标掩码
        :param iou: 2个BBox的iou [batch, 7, 7, 2]
        :param response: [batch, 7, 7, 1]
        :return: 有目标掩码[batch, 7, 7, 2] 无目标掩码[batch, 7, 7, 2]
        '''
        # 计算各个cell各自所预测的几个边界框中的IOU的最大值
        object_mask = tf.reduce_max(iou, axis=-1, keep_dims=True)
        # 其维度为[batch_size, 7, 7, 2] 如果cell中真实有目标，那么该cell内iou最大的那个框的相应位置为1（就是负责预测该框），其余为0
        object_mask = tf.cast((iou >= object_mask), tf.float32)
        # 首先得出当前cell中负责进行目标预测的框，再与真实的置信度进行点乘，得出真实的包含有目标的cell中负责进行目标预测的框．
        object_mask = object_mask * response
        # 没有目标的框其维度为[batch_size, 7 , 7, 2]， 真实没有目标的区域都为1，真实有目标的区域为0
        no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        return object_mask, no_object_mask

