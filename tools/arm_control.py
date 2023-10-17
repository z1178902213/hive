from periphery import GPIO
import time


class Arm:
    def __init__(self):
        """
        初始化GPIO输入输出状态
        """
        self.gpio_in = GPIO(89, "in")
        self.gpio_out = GPIO(81, "out")

    def act(self):
        """
        激活机械臂
        """
        out_list = [False, True, False, True]
        for out in out_list:
            self.gpio_out.write(out)
            time.sleep(2)

    def close(self):
        """
        释放GPIO
        """
        self.gpio_in.close()
        self.gpio_out.close()
