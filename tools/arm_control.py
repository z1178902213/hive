from periphery import GPIO
import time


class Arm:
    def __init__(self, input=89, output=81):
        """
        初始化GPIO输入输出状态
        """
        self.gpio_in = GPIO(input, "in")
        self.gpio_out = GPIO(output, "out")

    def act(self, left=True):
        """
        激活机械臂，左下格发送01，右下格发送10，间隔5秒

        Keyword Arguments:
            left {bool} -- 是否是左下格的六边形 (default: {True})
        """
        if left:
            out_list = [False, True]
            print("抓取左下幼虫")
        else:
            out_list = [True, False]
            print("抓取右下幼虫")
        for out in out_list:
            self.gpio_out.write(out)
            time.sleep(0.5)

    def receive_signal(self):
        """
        接收机械臂发送的信号

        Returns:
            bool -- True为1，False为0
        """
        return self.gpio_in.read()

    def close(self):
        """
        释放GPIO
        """
        self.gpio_in.close()
        self.gpio_out.close()
