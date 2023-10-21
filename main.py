from tools.common import Clock, find_and_check_cameras, load_config

clock = Clock()
from threading import Thread
import os
import cv2
from tools.find_contour import FindContour
import warnings
from tools.robort import Arm, Robort
from tools.yolo_process import *
from tools.find_worm import *

# 关闭烦人的tensorflow的FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

# 全局变量
RKNN_MODEL = "worm.rknn"
BOX_THRESH = 0.5
NMS_THRESH = 0.0
IMG_SIZE = 640
RESHAPE_RATIO = 3  # 在进行角点检测的时候所进行的放大比例
CAMERA_LEFT = "./test.mp4"  # 视频源，或者说是摄像头
CLASSES = "worm"
SAVE_IMG = True


def run(rk_yolo, camera_list, config):
    # 解析配置
    output_root = config["outputRoot"]
    problem_root = config["problemRoot"]
    gpio_pin = config["gpioPin"]
    gpio_map = config["gpioMap"]
    running_mode = config["runningMode"]
    multiple_camera = config['multipleCamera']

    # 初始化输出文件夹
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(problem_root):
        os.makedirs(problem_root)

    # 初始化摄像头设备
    roborts = []
    # 若开启多摄像头，则将摄像头全部都读取进去
    if multiple_camera:
        for index, camera_id in enumerate(camera_list):
            if index == len(gpio_pin):
                print(f'没有这么多组GPIO，请保持摄像头数量在{len(gpio_pin)}个')
            roborts.append(Robort(rk_yolo, camera_id, gpio_pin[index], gpio_map))
    else:
        roborts.append(Robort(rk_yolo, camera_list[0], gpio_pin[0], gpio_map))

    # 进入主程序逻辑
    while True:
        for index, bot in enumerate(roborts):
            if bot.capture():
                img = bot.draw()
                cv2.imshow(f"{index}", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print('--> 没有捕获到图片...')



if __name__ == "__main__":
    print("--> 加载配置...", end='')
    config = load_config("./config.json")  # 加载配置文件
    print("加载完成")

    # 初始化
    print("--> 初始化RKNN环境...", end='')
    rk_yolo = RK_YOLO(RKNN_MODEL)
    clock.print_time("成功")

    # camera_list = find_and_check_cameras()
    camera_list = ['test1.mp4', 'test2.mp4']
    print(f'--> 检测到{len(camera_list)}个摄像头，单摄像头模式下默认使用第1个')
    run(rk_yolo, camera_list, config)
