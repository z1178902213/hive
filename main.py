"""
@File     : main.py
@Time     : 2023/9/24 21:01
@Author   : 张伟杰
@Func     : 使用rknn模型检测幼虫
"""
import time

# 全局变量
RKNN_MODEL = "worm.rknn"
BOX_THRESH = 0.5
NMS_THRESH = 0.0
IMG_SIZE = 640
RESHAPE_RATIO = 3  # 在进行角点检测的时候所进行的放大比例
IMAGE_FOLDER = "./all_images"
OUTPUTS_ROOT = IMAGE_FOLDER + "_outputs"
PROBLEM_ROOT = "./problems"
CLASSES = "worm"


# 创建一个时钟类，用来计时
class Clock:
    def __init__(self):
        self.start = time.time()

    def cal_interval_time(self):
        self.interval_time = time.time() - self.start

    def print_time(self, str=None, restart=True):
        self.cal_interval_time()
        if str:
            print(f"{str}，耗时{self.interval_time:.2f}s\n")
        else:
            print(f"耗时{self.interval_time:.2f}s\n")
        if restart:
            self.start = time.time()


clock = Clock()

print("加载依赖文件...")
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from modules.find_contour import FindContour
import numpy as np
import cv2
from rknn.api import RKNN
import warnings

clock.print_time("加载依赖文件完成")

# 关闭烦人的tensorflow的FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


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


def cone_curve(x, a, b, c):
    """
    二维圆锥曲线，用于拟合一个圆

    Arguments:
        x {float} -- 输入的点的坐标
        a {float} -- 圆的参数a
        b {float} -- 圆的参数b
        c {float} -- 圆的参数c

    Returns:
        float -- 圆的坐标对应参数的值
    """
    return a * x**2 + b * x + c


def circle_residuals(params, x, y):
    """
    对圆进行拟合的优化函数

    Arguments:
        params {float} -- 包含圆心坐标(a, b)和半径r
        x {float} -- x的值
        y {float} -- y的值

    Returns:
        float -- 预测的圆的半径与实际半径的差值
    """
    # params 包含圆心坐标 (a, b) 和半径 r
    a, b, r = params
    return np.sqrt((x - a) ** 2 + (y - b) ** 2) - r


def fast_ratio(image, ratio):
    """
    对图像按比例放大后进行角点检测，返回关键点坐标

    Arguments:
        image {ndarray} -- 图像数据

    Returns:
        list -- 检测的fast角点关键点的坐标
    """
    h, w, _ = image.shape
    img = cv2.resize(image.copy(), (w * ratio, h * ratio))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行 FAST 角点检测
    fast_detect = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
    key_points = fast_detect.detect(gray, None)
    resume_points = []

    # 将比例进行缩放
    for keypoint in key_points:
        x, y = keypoint.pt[0] / ratio, keypoint.pt[1] / ratio
        resume_points.append((x, y))
    return resume_points


def fit_circle(key_points):
    x_data = []
    y_data = []
    # 获取所有的角点
    for keypoint in key_points:
        x, y = keypoint
        x_data.append(x)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    initial_a = np.mean(x_data)
    initial_b = np.mean(y_data)
    distances = np.sqrt((x_data - initial_a) ** 2 + (y_data - initial_b) ** 2)
    initial_r = np.mean(distances)
    # 初始化拟合参数（圆心坐标和半径）
    initial_params = [initial_a, initial_b, initial_r]  # 用你的估计值替换这些参数
    # 利用least_squares函数拟合圆
    result = least_squares(circle_residuals, initial_params, args=(x_data, y_data))

    # 返回拟合结果
    return result.x


def draw_circle(
    img, circle_params, standard, offsets, thickness=10, draw_keypoints=False
):
    a, b, r = circle_params
    offset_x, offset_y = offsets
    # 生成一组用于绘制圆的角度值
    theta = np.linspace(0, 2 * np.pi, 100)
    x_values = a + r * np.cos(theta) + offset_x
    y_values = b + r * np.sin(theta) + offset_y

    # 将坐标变为整数
    x_values = x_values.astype(int)
    y_values = y_values.astype(int)

    # 绘制原始图像
    draw_color = (0, 255, 0) if (r / 3 * 2 / standard < 2.5) else (0, 0, 255)

    if draw_keypoints:
        for point in zip(x_values, y_values):
            cv2.circle(img, (point[0], point[1]), 3, (0, 0, 255), -1)  # 标记原始散点

    # 绘制拟合的曲线
    for i in range(len(x_values) - 1):
        cv2.line(
            img,
            (x_values[i], y_values[i]),
            (x_values[i + 1], y_values[i + 1]),
            draw_color,
            thickness,
        )

    return img, int(r / 3)


def box_resume(boxes, ratio, padding):
    """
    恢复被letterbox处理后的检测框位置与大小
    """
    ret = []
    for box in boxes:
        left, top, right, bottom = box
        left = (left - padding[0]) / ratio[0]
        right = (right - padding[0]) / ratio[0]
        top = (top - padding[1]) / ratio[0]
        bottom = (bottom - padding[1]) / ratio[0]
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        ret.append((left, top, right, bottom))
    return ret


if __name__ == "__main__":
    if not os.path.exists(OUTPUTS_ROOT):
        os.makedirs(OUTPUTS_ROOT)
    if not os.path.exists(PROBLEM_ROOT):
        os.makedirs(PROBLEM_ROOT)
    # step1: 加载RKNN模型
    print("--> step1: 加载RKNN模型...")
    rknn = RKNN()
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("--> 加载模型失败，程序终止")
        exit(ret)
    clock.print_time("--> 加载模型成功")

    # step2: 初始化RKNN运行环境
    print("--> step2: 初始化RKNN运行环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("--> 初始化RKNN运行环境失败，程序终止")
        exit(ret)
    clock.print_time("--> 初始化RKNN运行环境成功")

    print("--> step3: 读取全部图像并逐个进行识别...")
    val_clock = Clock()
    worm_num = 0
    worm_inside_num = 0
    images = os.listdir(IMAGE_FOLDER)
    for image_name in images:
        SOURCE = f"{IMAGE_FOLDER}/{image_name}"
        print(f"--> 处理图像{SOURCE}...", end="")
        save_dir = f'{OUTPUTS_ROOT}/{image_name.split(".")[0]}_detect.jpg'

        img = cv2.imread(SOURCE)
        h, w, c = img.shape  # 保存图像的高、宽、通道数

        img_letterbox, ratio, (dw, dh) = letterbox(
            img.copy(), new_shape=(IMG_SIZE, IMG_SIZE)
        )
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        outputs = rknn.inference(inputs=[img_rgb])
        input_data = list()
        input_data.append(
            np.transpose(outputs[0].reshape([3, 80, 80, 6]), (1, 2, 0, 3))
        )
        input_data.append(
            np.transpose(outputs[1].reshape([3, 40, 40, 6]), (1, 2, 0, 3))
        )
        input_data.append(
            np.transpose(outputs[2].reshape([3, 20, 20, 6]), (1, 2, 0, 3))
        )
        boxes, classes, scores = yolov5_post_process(input_data)
        if boxes is not None:
            boxes = box_resume(boxes, ratio, (dw, dh))
        else:
            problem_dir = f'{PROBLEM_ROOT}/{image_name.split(".")[0]}_nobox_problem.jpg'
            cv2.imwrite(problem_dir, img)
            print(f"没有检测到幼虫，跳过该图片")
            continue
        worm_num += len(boxes)

        # 实例化六边形框检测对象
        my_find = FindContour(
            img, 2, True, False, doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30)
        )
        if my_find.standard2 <= 0:
            problem_dir = (
                f'{PROBLEM_ROOT}/{image_name.split(".")[0]}_standard2_problem.jpg'
            )
            cv2.imwrite(problem_dir, img)
            print(f"my_find.standard2 <= 0，跳过该图片")
            continue
        # 对所有检测框进行判断
        for index, xyxy in enumerate(boxes):
            cut_image = img[int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])]
            cut_image_h, cut_image_w, cut_image_c = cut_image.shape
            try:
                fast_keypoints = fast_ratio(cut_image, RESHAPE_RATIO)
                circle = fit_circle(fast_keypoints)
                # 为中心下方的两个六边形绘制圆与标签
                if my_find.in_contour(xyxy):
                    worm_inside_num += 1
                    draw_circle(
                        img,
                        circle,
                        my_find.standard2,
                        (int(xyxy[0]), int(xyxy[1])),
                        thickness=2,
                    )
                    text_scale = ((h / 1080) + (w / 1920)) / 2
                    cv2.putText(
                        img,
                        f"{(circle[2] * 2 / my_find.standard2):.2f}mm",
                        (int(xyxy[0]), int(xyxy[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        (0, 0, 255),
                        2,
                    )
            except Exception as e:
                problem_dir = (
                    f'{PROBLEM_ROOT}/{image_name.split(".")[0]}_unknown_problem.jpg'
                )
                cv2.imwrite(problem_dir, img)
                print(f"未知错误，保存该图片在{problem_dir}，跳过该图片")
                continue
        cv2.imwrite(save_dir, img)
        clock.print_time(f"处理完成")

    val_clock.cal_interval_time()
    print(
        f"\n--> 一共处理了{len(images)}张图片\n耗时{val_clock.interval_time:.2f}s\n平均一秒处理{len(images) / val_clock.interval_time:.2f}张图片"
    )
    print(
        f"\n--> 一共识别了{worm_num}只幼虫\n平均一秒识别{worm_num / val_clock.interval_time:.2f}只幼虫"
    )
    print(
        f"\n--> 一共计算了{worm_inside_num}只幼虫的长度\n平均一秒计算{worm_inside_num / val_clock.interval_time:.2f}只幼虫"
    )
