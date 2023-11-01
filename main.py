from tools.common import Clock, load_config, find_and_check_cameras

clock = Clock()
import os
import cv2
import warnings
from tools.robort import Robort
from tools.yolo_process import *
from tools.find_worm import *

# 关闭烦人的tensorflow的FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)



def run(rk_yolo, camera_list, config):
    # 解析配置
    output_root = config["outputRoot"]
    problem_root = config["problemRoot"]
    gpio_pin = config["gpioPin"]
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
            roborts.append(Robort(rk_yolo, camera_id, index, config))
    else:
        roborts.append(Robort(rk_yolo, camera_list[0], 0, config))
    
    count = 0
    # 进入主程序逻辑
    while True:
        for index, bot in enumerate(roborts):
            if bot.capture():
                try:
                    img = bot.draw()
                except Exception as e:
                    print(f'error: 未知错误，返回原图，错误日志如下：\n{e}\n')
                    img = bot.image
                if isinstance(img, tuple):
                    img, worm_loc = img
                    bot.catch(worm_loc)
                # bot.out.write(img)
                cv2.namedWindow(f'{index}', cv2.WINDOW_KEEPRATIO)
                cv2.imshow(f"{index}", img)
                # cv2.imwrite(f'./camera_output/{index}_{count}.jpg', img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print('--> 没有捕获到图片...')
        count += 1


if __name__ == "__main__":
    print("--> 加载配置...", end='')
    config = load_config("./config.json")  # 加载配置文件
    print("加载完成")

    # 初始化
    print("--> 初始化RKNN环境...", end='')
    rk_yolo = RK_YOLO("./worm.rknn")
    clock.print_time("成功")

    camera_list = find_and_check_cameras()
    # camera_list = ['test1.mp4', 'test2.mp4']
    tips = '' if config['multipleCamera'] else '，未开启多摄像头模式，默认使用第一个摄像头'
    print(f'--> 检测到{len(camera_list)}个摄像头{tips}')
    if camera_list:
        run(rk_yolo, camera_list, config)
