from tools.common import Clock, find_and_check_cameras, load_config

clock = Clock()
from threading import Thread
import os
import cv2
from tools.find_contour import FindContour
import warnings
from tools.arm_control import Arm
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


def run(rk_yolo, camera_id, config):
    # 解析配置
    output_root = config["output_root"]
    problem_root = config["problem_root"]
    gpio_pin = config["gpio_pin"]
    gpio_map = config["gpio_map"]
    gpio_in = gpio_map[str(gpio_pin[0])]
    gpio_out1 = gpio_map[str(gpio_pin[1])]
    gpio_out2 = gpio_map[str(gpio_pin[2])]
    running_mode = config["running_mode"]

    # 初始化输出文件夹
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(problem_root):
        os.makedirs(problem_root)

    # 初始化摄像头设备
    cap = cv2.VideoCapture(camera_id)

    # 初始化机械臂
    arm = Arm(gpio_in,gpio_out1,gpio_out2)

    # 进入主程序逻辑
    count = 0
    is_wait = False
    while True:
        if running_mode == 0:  # 校准模式
            ret, frame = cap.read()
            if ret:
                h, w, c = frame.shape
                my_find = FindContour(
                    frame,
                    2,
                    True,
                    False,
                    doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30),
                )
                cv2.putText(
                    frame,
                    f"GPIO {gpio_pin[0]} {gpio_pin[1]} {gpio_pin[2]}",
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(f"{index}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        elif running_mode == 1:  # 识别模式
            if arm.receive_signal():
                is_wait = False
                ret, frame = cap.read()
                if ret:
                    count += 1
                    print(f"--> 处理第{count}帧...", end="")
                    save_dir = f"{output_root}/{count}_detect.jpg"
                    h, w, c = frame.shape  # 保存帧的高、宽、通道数

                    frame_letterbox, ratio, (dw, dh) = letterbox(
                        frame.copy(), new_shape=(IMG_SIZE, IMG_SIZE)
                    )
                    frame_rgb = cv2.cvtColor(frame_letterbox, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))

                    boxes, classes, scores = rk_yolo.detect(
                        frame_rgb, IMG_SIZE, BOX_THRESH, NMS_THRESH
                    )

                    if boxes is not None:
                        boxes = box_resume(boxes, ratio, (dw, dh))
                    else:
                        problem_dir = f"{problem_root}/{count}_nobox_problem.jpg"
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
                        problem_dir = f"{problem_root}/{count}_standard2_problem.jpg"
                        cv2.imwrite(problem_dir, frame)
                        print(f"my_find.standard2 <= 0，跳过该帧")
                        continue
                    # 对所有检测框进行判断
                    for xyxy in boxes:
                        cut_image = frame[
                            int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
                        ]
                        try:
                            fast_keypoints = fast_ratio(cut_image, RESHAPE_RATIO)
                            circle = fit_circle(fast_keypoints)
                            # 为中心下方的两个六边形绘制圆与标签
                            if my_find.in_contour(xyxy):
                                # left_arm.act(True)
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
                            problem_dir = f"{problem_root}/{count}_unknown_problem.jpg"
                            cv2.imwrite(problem_dir, frame)
                            print(f"未知错误，保存该图片在{problem_dir}，跳过该帧")
                            continue
                    if SAVE_IMG:
                        cv2.imwrite(save_dir, frame)
                    clock.print_time(f"处理完成")
                    ret, frame = cap.read()
                else:
                    print("--> 未获取到视频帧，请检查摄像头是否插好")
            else:
                if not is_wait:
                    print("--> 等待机械臂信号...")
                    is_wait = True
        else:
            print("--> 运行模式有误，0为校准模式，1为识别模式，无其他模式，请修改配置文件config.json中的running_mode参数")


if __name__ == "__main__":
    double_camera = False

    print("--> 加载配置")
    config = load_config("./config.json")  # 加载配置文件

    # 初始化
    print("--> 初始化RKNN环境...")
    rk_yolo = RK_YOLO(RKNN_MODEL)
    clock.print_time("--> 初始化RKNN环境成功")

    camera_list = find_and_check_cameras()
    run(rk_yolo, camera_list, config)
