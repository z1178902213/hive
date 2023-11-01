from tools.common import Clock

clock = Clock()

print("--> step0: 加载依赖文件...")
import os
import cv2
import numpy as np
from tools.find_contour import FindContour
import cv2
from rknnlite.api import RKNNLite as RKNN
import warnings
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
IMAGE_FOLDER = "./test_image"
OUTPUTS_ROOT = IMAGE_FOLDER + "_outputs"
PROBLEM_ROOT = "./problems"
CLASSES = "worm"


if __name__ == "__main__":
    # 创建运行日志文件夹
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
        boxes, classes, scores = yolov5_post_process(
            input_data,
            image_size=IMG_SIZE,
            box_thresh=BOX_THRESH,
            nms_thresh=NMS_THRESH,
        )
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
            img, 2, False, True, doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30)
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
            left, top, right, bottom = xyxy
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            if right > w:
                right = w
            if bottom > h:
                bottom = h
            cut_image = img[int(top) : int(bottom), int(left) : int(right)]
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
                        1 if w < 1920 else 2,
                    )
            except Exception as e:
                problem_dir = (
                    f'{PROBLEM_ROOT}/{image_name.split(".")[0]}_unknown_problem.jpg'
                )
                cv2.imwrite(problem_dir, img)
                print(f"未知错误，保存该图片在{problem_dir}，跳过该图片，错误日志：\n{e}\n")
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
