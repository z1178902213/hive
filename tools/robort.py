from periphery import GPIO
from tools.find_contour import FindContour
from tools.yolo_process import letterbox, box_resume
from tools.find_worm import *
import cv2
import time

class Robort:
    def __init__(self, rk_yolo, camera_id, gpio_pin, gpio_map):
        self.rk_yolo = rk_yolo
        self.gpio_pin = gpio_pin
        self.eye = cv2.VideoCapture(camera_id)
        self.arm = Arm(gpio_pin, gpio_map)
        self.image = None
        self.mode = 0
        self.circle=False
        self.doji=True
        self.center=True
        self.gpio=True
    
    def change_mode(self, mode):
        """
        变换模式

        Arguments:
            mode {int} -- 模式编号 0校准 1检测
        """
        self.mode = mode
        if mode == 0:
            self.circle=False
            self.doji=True
            self.center=True
            self.gpio=True
        elif mode == 1:
            self.circle=True
            self.doji=True
            self.center=False
            self.gpio=False
    
    def capture(self):
        """
        捕获一张图像

        Returns:
            bool -- 是否捕获成功
        """
        ret, self.image = self.eye.read()
        return ret

    def draw(self):
        if self.mode == 0:
            h, w, c = self.image.shape
            FindContour(
                self.image,
                2,
                draw_circle=self.circle,
                is_draw_doji=self.doji,
                doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30),
                is_draw_center=self.center,
            )
            if self.gpio:
                cv2.putText(
                    self.image,
                    f"GPIO {self.gpio_pin[0]} {self.gpio_pin[1]} {self.gpio_pin[2]}",
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            return self.image
        elif self.mode == 1:
            h, w, c = self.image.shape  # 保存帧的高、宽、通道数

            frame_letterbox, ratio, (dw, dh) = letterbox(
                self.image.copy(), new_shape=(640, 640)
            )
            frame_rgb = cv2.cvtColor(frame_letterbox, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 640))

            boxes, classes, scores = self.rk_yolo.detect(frame_rgb, 640, 0.5, 0)

            if boxes is not None:
                boxes = box_resume(boxes, ratio, (dw, dh))
            else:
                print(f"--> 没有检测到幼虫...")
                return self.image

            # 实例化六边形框检测对象
            my_find = FindContour(
                self.image,
                2,
                draw_circle=self.circle,
                is_draw_doji=self.doji,
                doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30),
                is_draw_center=self.center,
            )
            if my_find.standard2 <= 0:
                print(f"--> 长度估计出错...")
                return self.image
            # 对所有检测框进行判断
            for xyxy in boxes:
                cut_image = self.image[
                    int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
                ]
                try:
                    fast_keypoints = fast_ratio(cut_image, 3)
                    circle = fit_circle(fast_keypoints)
                    # 为中心下方的两个六边形绘制圆与标签
                    if my_find.in_contour(xyxy):
                        # left_arm.act(True)
                        draw_circle(
                            self.image,
                            circle,
                            my_find.standard2,
                            (int(xyxy[0]), int(xyxy[1])),
                            thickness=2,
                        )
                        cv2.putText(
                            self.image,
                            f"{(circle[2] * 2 / my_find.standard2):.2f}mm",
                            (int(xyxy[0]), int(xyxy[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            ((h / 1080) + (w / 1920)) / 2,
                            (0, 0, 255),
                            2,
                        )
                except Exception:
                    print(f"--> 未知错误，跳过")
                    return self.image
        else:
            cv2.putText(
                self.image,
                f"目前只有0和1的模式，请修改config.json中的配置",
                (int(self.image.shape[1] / 2), int(self.image.shape[0] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                (((h / 1080) + (w / 1920)) / 2) * 3,
                (0, 0, 255),
                4,
            )
            return self.image

    def catch(self):
        """
        抓幼虫
        """
        pass


class Arm:
    def __init__(self, gpio_pin, gpio_map):
        """
        初始化GPIO

        Arguments:
            gpio_pin {list} -- GPIO列表 对应输入 输出1 输出2
            gpio_map {dict} -- GPIO字典 逻辑GPIO和实际GPIO号的对应
        """
        input = gpio_map[str(gpio_pin[0])]
        output0 = gpio_map[str(gpio_pin[1])]
        output1 = gpio_map[str(gpio_pin[2])]
        self.gpio_in = GPIO(input, "in")
        self.gpio_out0 = GPIO(output0, "out")
        self.gpio_out1 = GPIO(output1, "out")

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
        state = self.gpio_in.read()
        return state

    def close(self):
        """
        释放GPIO
        """
        self.gpio_in.close()
        self.gpio_out.close()
