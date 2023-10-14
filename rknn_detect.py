import time


class Clock:
    def __init__(self):
        self.start = time.time()

    def print_time(self, str=None, restart=True):
        interval_time = time.time() - self.start
        if str:
            print(f"{str}，耗时{interval_time:.2f}s\n")
        else:
            print(f"耗时{interval_time:.2f}s\n")
        if restart:
            self.start = time.time()


clock = Clock()

print("加载依赖文件...")
import numpy as np
import cv2
from rknn.api import RKNN
import warnings

clock.print_time("加载依赖文件完成")

warnings.simplefilter(
    action="ignore", category=FutureWarning
)  # 关闭烦人的tensorflow的FutureWarning

CLASSES = "worm"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2]) * 2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE / grid_h)

    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!
    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.
    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_confidences[..., 0] >= BOX_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [
        [10, 13],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
        [116, 90],
        [156, 198],
        [373, 326],
    ]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes, ratio, padding):
    """Draw the boxes on the image.
    # Argument:
        image: 原始图像
        boxes: ndarray, 对象检测框
        classes: ndarray, 目标类别
        scores: ndarray, 目标检测分数
        ratio: tuple, 缩放比例
        padding: tuple, 填充宽度与高度(dw, dh)
    """
    for box, score, cl in zip(boxes, scores, classes):
        left, top, right, bottom = box
        print(box)
        left = (left - padding[0]) / ratio[0]
        right = (right - padding[0]) / ratio[0]
        top = (top - padding[1]) / ratio[0]
        bottom = (bottom - padding[1]) / ratio[0]
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            "{0} {1:.2f}".format(CLASSES[cl], score),
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


if __name__ == "__main__":
    RKNN_MODEL = "worm.rknn"
    IMG_PATH = "./test.jpg"
    BOX_THRESH = 0.5
    NMS_THRESH = 0.45
    IMG_SIZE = 640

    print("--> step1: 加载RKNN模型...")
    rknn = RKNN()
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("--> 加载模型失败，程序终止")
        exit(ret)
    clock.print_time("--> 加载模型成功")

    print("--> step2: 初始化RKNN运行环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("--> 初始化RKNN运行环境失败，程序终止")
        exit(ret)
    clock.print_time("--> 初始化RKNN运行环境成功")

    print("--> step3: 读取图像并进行图像预处理...")
    img = cv2.imread(IMG_PATH)
    img_letterbox, ratio, (dw, dh) = letterbox(
        img.copy(), new_shape=(IMG_SIZE, IMG_SIZE)
    )
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    # img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    clock.print_time("--> 处理完成")

    print("--> step4: 开始推理...")
    outputs = rknn.inference(inputs=[img_rgb])
    clock.print_time("--> 推理完成")

    print("--> step5: 处理推理数据...")
    input_data = list()
    input_data.append(np.transpose(outputs[0].reshape([3, 80, 80, 6]), (1, 2, 0, 3)))
    input_data.append(np.transpose(outputs[1].reshape([3, 40, 40, 6]), (1, 2, 0, 3)))
    input_data.append(np.transpose(outputs[2].reshape([3, 20, 20, 6]), (1, 2, 0, 3)))
    boxes, classes, scores = yolov5_post_process(input_data)
    clock.print_time("--> 处理完成")

    print("--> step6: 绘制检测框...")
    if boxes is not None:
        draw(img, boxes, scores, classes, ratio, (dw, dh))
    cv2.imwrite("result.jpg", img)
    rknn.release()
    clock.print_time("--> 绘制完成")
