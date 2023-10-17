"""
@File     : main.py
@Time     : 2023/10/17 16:08
@Author   : 张伟杰
@Func     : 使用RKNN模型检测双摄像头获取到的图像，并收发GPIO信号
"""
from tools.clock import Clock

clock = Clock()

print("--> step0: 加载依赖文件...")
import os
import cv2
import numpy as np
from tools.find_contour import FindContour
from rknn.api import RKNN
import warnings
from tools.arm_control import Arm
from tools.yolo_process import *
from tools.find_worm import *

clock.print_time("--> 加载依赖文件完成")

# 关闭烦人的tensorflow的FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# 全局变量
RKNN_MODEL = "worm.rknn"
BOX_THRESH = 0.5
NMS_THRESH = 0.0
IMG_SIZE = 640
RESHAPE_RATIO = 3  # 在进行角点检测的时候所进行的放大比例
CAMERA_LEFT = "./test.mp4"  # 视频源，或者说是摄像头
OUTPUTS_ROOT = "camera_outputs"
PROBLEM_ROOT = "./problems"
CLASSES = "worm"
SAVE_IMG = True


if __name__ == "__main__":
    flag = True
    # 创建运行日志文件夹
    if not os.path.exists(OUTPUTS_ROOT):
        os.makedirs(OUTPUTS_ROOT)
    if not os.path.exists(PROBLEM_ROOT):
        os.makedirs(PROBLEM_ROOT)

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

    print("--> step3: 读取摄像头并进行识别...")
    val_clock = Clock()
    left_cap = cv2.VideoCapture(CAMERA_LEFT)
    left_arm = Arm(89, 81)
    while flag:
        while not left_arm.receive_signal():
            ret, frame = left_cap.read()
            count = 0
            if ret:
                count += 1
                print(f"--> 处理第{count}帧...", end="")
                save_dir = f"{OUTPUTS_ROOT}/{count}_detect.jpg"
                h, w, c = frame.shape  # 保存帧的高、宽、通道数

                frame_letterbox, ratio, (dw, dh) = letterbox(
                    frame.copy(), new_shape=(IMG_SIZE, IMG_SIZE)
                )
                frame_rgb = cv2.cvtColor(frame_letterbox, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))

                outputs = rknn.inference(inputs=[frame_rgb])
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
                boxes, classes, scores = yolov5_post_process(
                    input_data,
                    image_size=IMG_SIZE,
                    box_thresh=BOX_THRESH,
                    nms_thresh=NMS_THRESH,
                )
                if boxes is not None:
                    boxes = box_resume(boxes, ratio, (dw, dh))
                else:
                    problem_dir = f"{PROBLEM_ROOT}/{count}_nobox_problem.jpg"
                    cv2.imwrite(problem_dir, frame)
                    print(f"没有检测到幼虫，跳过该帧")
                    continue

                # 实例化六边形框检测对象
                my_find = FindContour(
                    frame,
                    2,
                    True,
                    False,
                    doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30),
                )
                if my_find.standard2 <= 0:
                    problem_dir = f"{PROBLEM_ROOT}/{count}_standard2_problem.jpg"
                    cv2.imwrite(problem_dir, frame)
                    print(f"my_find.standard2 <= 0，跳过该帧")
                    continue
                # 对所有检测框进行判断
                for index, xyxy in enumerate(boxes):
                    cut_image = frame[
                        int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
                    ]
                    cut_image_h, cut_image_w, cut_image_c = cut_image.shape
                    try:
                        fast_keypoints = fast_ratio(cut_image, RESHAPE_RATIO)
                        circle = fit_circle(fast_keypoints)
                        # 为中心下方的两个六边形绘制圆与标签
                        if my_find.in_contour(xyxy):
                            left_arm.act(True)
                            draw_circle(
                                frame,
                                circle,
                                my_find.standard2,
                                (int(xyxy[0]), int(xyxy[1])),
                                thickness=2,
                            )
                            text_scale = ((h / 1080) + (w / 1920)) / 2
                            cv2.putText(
                                frame,
                                f"{(circle[2] * 2 / my_find.standard2):.2f}mm",
                                (int(xyxy[0]), int(xyxy[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                text_scale,
                                (0, 0, 255),
                                2,
                            )
                    except Exception as e:
                        problem_dir = f"{PROBLEM_ROOT}/{count}_unknown_problem.jpg"
                        cv2.imwrite(problem_dir, frame)
                        print(f"未知错误，保存该图片在{problem_dir}，跳过该帧")
                        continue
                if SAVE_IMG:
                    cv2.imwrite(save_dir, frame)
                clock.print_time(f"处理完成")
                ret, frame = left_cap.read()
            else:
                print("--> 未获取到视频帧，请检查摄像头是否插好")
    val_clock.cal_interval_time()
    print(
        f"\n--> 一共处理了{count}帧图像\n耗时{val_clock.interval_time:.2f}s\n平均一秒处理{count / val_clock.interval_time:.2f}帧图像"
    )
