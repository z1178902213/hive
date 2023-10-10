"""
@File     : test.py
@Time     : 2023/9/24 21:01
@Author   : 张伟杰
@Func     :
"""
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from detect import parse_opt as yolov5_parse_opt
from detect import run as yolov5_run
from modules.find_contour import FindContour

# 定义拟合的曲线函数
WEIGHTS = '.\\worm.pt'
SOURCE = '.\\1.jpg'
IMGSZ = (640, 640)
DATA = '.\\bee_children_v1\\data.yaml'
SHOW_CUT_IMAGE = False
SHOW_FAST_CUT_IMAGE = False
DRAW_KEYPOINTS = False
IMAGES_ROOT = './all_images'
OUTPUTS_ROOT = IMAGES_ROOT + '_outputs3'
RESHAPE_RATIO = 3       # 在进行角点检测的时候所进行的放大比例


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
    return a * x ** 2 + b * x + c


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
    fast_detect = cv2.FastFeatureDetector_create(
        threshold=10, nonmaxSuppression=True)
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
    result = least_squares(
        circle_residuals, initial_params, args=(x_data, y_data))

    # 返回拟合结果
    return result.x


def draw_circle(img, circle_params, standard, offsets, thickness=10):
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

    if DRAW_KEYPOINTS:
        for point in zip(x_values, y_values):
            cv2.circle(img, (point[0], point[1]), 3, (0, 0, 255), -1)  # 标记原始散点

    # 绘制拟合的曲线
    for i in range(len(x_values) - 1):
        cv2.line(img, (x_values[i], y_values[i]),
                 (x_values[i + 1], y_values[i + 1]), draw_color, thickness)

    return img, int(r / 3)


if __name__ == '__main__':
    images = os.listdir(IMAGES_ROOT)
    for image_name in images:
        SOURCE = f'{IMAGES_ROOT}/{image_name}'
        # 配置YOLOv5检测的参数
        opt = yolov5_parse_opt()
        opt.weights = WEIGHTS
        opt.source = SOURCE
        opt.imgsz = IMGSZ
        opt.data = DATA
        opt.save_txt = True
        boxes_list = yolov5_run(**vars(opt))
        # 读取图片
        image = cv2.imread(SOURCE)
        h, w, c = image.shape
        my_find = FindContour(image, 2, True, False,
                              doji_len=int((((h/1080)+(w/1920))/2) * 30))
        if my_find.standard2 <= 0:
            continue
        if not os.path.exists(OUTPUTS_ROOT):
            os.makedirs(OUTPUTS_ROOT)
        # 对所有检测框进行判断
        for box_list in boxes_list:
            for index, box in enumerate(box_list):
                xyxy, conf = box
                cut_image = image[int(xyxy[1]):int(
                    xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                cut_image_h, cut_image_w, cut_image_c = cut_image.shape
                if SHOW_CUT_IMAGE:
                    cv2.imwrite(
                        f'{OUTPUTS_ROOT}/{SOURCE.split(".")[1]}_cut_{index}.jpg', cut_image)
                try:
                    fast_keypoints = fast_ratio(
                        cut_image, RESHAPE_RATIO)
                    circle = fit_circle(fast_keypoints)
                    # 为中心下方的两个六边形绘制圆与标签
                    if my_find.in_contour(xyxy):
                        draw_circle(image, circle, my_find.standard2,
                                    (int(xyxy[0]), int(xyxy[1])), thickness=2)
                        text_scale = ((h/1080)+(w/1920))/2
                        cv2.putText(image, f'{(circle[2] * 2 / my_find.standard2):.2f}mm', (int(
                            xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 2)
                except Exception as e:
                    print(f'未知错误{e}，跳过该box')
                    continue
            cv2.imwrite(
                f'{OUTPUTS_ROOT}/{image_name.split(".")[0]}_detect.jpg', image)
