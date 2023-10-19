import cv2
import time
import json


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def find_and_check_cameras():
    # 设置cv日志等级，忽视烦人的WARNING和ERROR
    log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)
    ok_list = []
    for i in range(0, 20):  # 遍历摄像头编号，找到能用的双目摄像头
        cap = cv2.VideoCapture(i)
        ret, _ = cap.read()
        if ret:
            ok_list.append(i)
            cap.release()
    cv2.setLogLevel(log_level)  # 恢复日志等级
    return ok_list


# 创建一个时钟类，用来计时
class Clock:
    def __init__(self):
        self.start = time.time()

    def cal_interval_time(self):
        """
        计算间隔时间
        """
        self.interval_time = time.time() - self.start

    def print_time(self, str=None, restart=True):
        """
        打印输出信息，并输出间隔时间

        Keyword Arguments:
            str {str} -- 需要打印输出的文本 (default: {None})
            restart {bool} -- 是否重新开始计时 (default: {True})
        """
        self.cal_interval_time()
        if str:
            print(f"{str}，耗时{self.interval_time:.2f}s\n")
        else:
            print(f"耗时{self.interval_time:.2f}s\n")
        if restart:
            self.start = time.time()
