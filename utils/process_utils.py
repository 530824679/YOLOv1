import cv2
import numpy as np
from cfg.config import model_params, path_params, test_params, CLASSES

def calculate_iou(box_1, box_2):
    """
    calculate iou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of iou
    """
    # calculate area of each box
    area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_2 = (box_2[2] - box_2[0]) * (box_1[3] - box_1[1])

    # find the edge of intersect box
    top = max(box_1[0], box_2[0])
    left = max(box_1[1], box_2[1])
    bottom = min(box_1[3], box_2[3])
    right = min(box_1[2], box_2[2])

    # if there is an intersect area
    if left >= right or top >= bottom:
        return 0

    # calculate the intersect area
    area_intersection = (right - left) * (bottom - top)

    # calculate the union area
    area_union = area_1 + area_2 - area_intersection

    iou = float(area_intersection) / area_union

    return iou

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def post_processing(outputs):
    """
    对网络的输出进行解析，通过类别置信度和非极大值抑制
    :param: outputs:网络的原始输出
    :return: 检测出的结果[box_num, x, y, w, h, prob]
    """
    boundary1 = model_params['cell_size'] * model_params['cell_size'] * model_params['num_classes']
    boundary2 = boundary1 + model_params['cell_size'] * model_params['cell_size'] * model_params['boxes_per_cell']

    predict_class_prob = outputs[0:boundary1]
    predict_class_prob = np.reshape(predict_class_prob, [model_params['cell_size'], model_params['cell_size'], model_params['num_classes']])
    # 解析出是否存在目标的置信度
    predict_confidence = outputs[boundary1:boundary2]
    predict_confidence = np.reshape(predict_confidence, [model_params['cell_size'], model_params['cell_size'], model_params['boxes_per_cell']])
    # 解析出bounding_box的参数信息，网络预测的bbox的中心坐标是相对于cell的偏移量
    predict_bboxs = outputs[boundary2:]
    predict_bboxs = np.reshape(predict_bboxs, [model_params['cell_size'], model_params['cell_size'], model_params['boxes_per_cell'], 4])

    # 将网络所预测的bbox相对于cell的偏移量转换为bbox的中心坐标在图像中的比例
    offset = np.array([np.arange(model_params['cell_size'])] * model_params['cell_size'] * model_params['boxes_per_cell'])
    offset = np.transpose(
        np.reshape(
            offset,
            [model_params['boxes_per_cell'], model_params['cell_size'], model_params['cell_size']]),
        (1, 2, 0))

    # 将中心坐标和宽，长转换为真实的像素值
    # 首先将偏移量形式的中心坐标和平方根形式的宽高转换为比例形式
    predict_bboxs[:, :, :, 0] += offset
    predict_bboxs[:, :, :, 1] += np.transpose(offset, (1, 0, 2))

    # 得到(x, y)相对于整张图片的位置比例
    predict_bboxs[:, :, :, :2] = 1.0 * predict_bboxs[:, :, :, 0:2] / model_params['cell_size']
    # 得到预测的宽度和高度乘以平方才能得到相对于整张图片的比例
    predict_bboxs[:, :, :, 2:] = np.square(predict_bboxs[:, :, :, 2:])
    # 得到相对于原图的坐标框
    predict_bboxs = predict_bboxs * model_params['image_size']

    # 计算得出cell中的各个预测框最终给出的概率值，prob=class_prob*confidence
    prob = np.zeros([model_params['cell_size'], model_params['cell_size'], model_params['boxes_per_cell'], model_params['num_classes']])
    for box in range(model_params['boxes_per_cell']):
        for class_n in range(model_params['num_classes']):
            prob[:, :, box, class_n] = predict_confidence[:, :, box] * predict_class_prob[:, :, class_n]

    # #如果大于prob_threshold，那么其对应的位置为true,反正false
    filter_probs = np.array(prob >= test_params['prob_threshold'], dtype='bool')
    # 找到为true的地方，用1来表示true, false是0
    filter_boxes = np.nonzero(filter_probs)

    # 找到符合的类别置信度
    probs_filtered = prob[filter_probs]
    boxes_filtered = predict_bboxs[filter_boxes[0], filter_boxes[1], filter_boxes[2]]
    # 若该cell类别置信度大于阈值，则只取类别置信度最大的那个框，一个cell只负责预测一个类别
    classes_num_filtered = np.argmax(
        filter_probs, axis=3)[
        filter_boxes[0], filter_boxes[1], filter_boxes[2]]

    # 类别置信度排序
    argsort = np.array(np.argsort(probs_filtered))[::-1]
    # 类别置信度排序
    boxes_filtered = boxes_filtered[argsort]
    # 找到符合条件的类别置信度，从大到小排序
    probs_filtered = probs_filtered[argsort]
    # 类别数过滤
    classes_num_filtered = classes_num_filtered[argsort]

    # 非极大值抑制算法
    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if calculate_iou(boxes_filtered[i], boxes_filtered[j]) > test_params['iou_threshold']:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    # 经过阈值和非极大值抑制之后得到的框
    boxes_filtered = boxes_filtered[filter_iou]
    # 经过阈值和非极大值抑制之后得到的类别置信度
    probs_filtered = probs_filtered[filter_iou]
    # 经过非极大值抑制之后得到的类别，一个cell只负责预测一个类别
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [CLASSES[classes_num_filtered[i]],
             boxes_filtered[i][0],
             boxes_filtered[i][1],
             boxes_filtered[i][2],
             boxes_filtered[i][3],
             probs_filtered[i]])

    return result

def show_results(image_path, results):
    image = cv2.imread(image_path).copy()

    if len(results) != 0:
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3] / 2)
            h = int(results[i][4] / 2)
            cv2.rectangle(image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA
            cv2.putText(image, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType)

    cv2.imwrite(path_params['test_output_dir'] + '/' + image_path.split('/')[-1], image)
