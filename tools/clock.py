import time


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
