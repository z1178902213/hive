from periphery import GPIO
from tools.find_contour import FindContour
from tools.yolo_process import letterbox, box_resume
from tools.find_worm import *
import cv2
import time
from threading import Thread

class Robort:
    def __init__(self, rk_yolo, camera_id, index, config):
        self.config = config
        self.index = index
        self.gpio_pin = self.config["gpioPin"][index]
        self.diameter_threshold = self.config['diameterThreshold']
        gpio_map = self.config["gpioMap"]
        
        self.rk_yolo = rk_yolo
        self.eye = cv2.VideoCapture(camera_id)
        self.arm = Arm(self.gpio_pin, gpio_map)
        self.image = None
        self.circle=True
        self.doji=True
        self.center=True
        self.gpio=True
        self.flag = True

    
    def capture(self):
        """
        捕获一张图像

        Returns:
            bool -- 是否捕获成功
        """
        ret, self.image = self.eye.read()
        if self.config['preProcess']:
            h, w, _ = self.image.shape
            # 对图像进行旋转
            self.image = cv2.warpAffine(self.image, cv2.getRotationMatrix2D((w / 2,h / 2), self.config['rotate'], 1), (w, h))
            # 对图像进行偏移
            self.image = cv2.warpAffine(self.image,np.float32([[1,0,self.config['leftOffset']],[0,1,self.config['topOffset']]]),(w,h))
        return ret

    def draw(self):
        # 绘制GPIO到图像左上角
        if self.gpio:
            cv2.putText(
                self.image,
                f"GPIO in:{self.gpio_pin[0]} out0:{self.gpio_pin[1]} out1:{self.gpio_pin[2]}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        origin = self.image.copy()
        h, w, c = self.image.shape  # 帧的高、宽、通道数
        
        # 进行letterbox操作
        frame_letterbox, ratio, (dw, dh) = letterbox(
            self.image.copy(), new_shape=(640, 640)
        )
        frame_rgb = cv2.cvtColor(frame_letterbox, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 640))

        # 使用yolo检测幼虫位置
        boxes, classes, scores = self.rk_yolo.detect(frame_rgb, 640, 0.5, 0)
        if boxes is not None:
            # 由于检测的图片是进行letterbox后的，这里需要将box恢复到原图像大小的相对位置
            boxes = box_resume(boxes, ratio, (dw, dh))
        else:
            self.print_info(f"没有检测到幼虫，返回原图...")
            return origin

        # 实例化六边形框检测对象
        try:
            my_find = FindContour(
                self.image,
                2,
                draw_circle=self.circle,
                is_draw_doji=self.doji,
                doji_len=int((((h / 1080) + (w / 1920)) / 2) * 30),
                is_draw_center=self.center,
                center_dis=self.config['dojiOffset']
            )
        except Exception as e:
            self.print_info(f"未知错误，返回原图，错误日志如下：\n{e}\n")
            return origin
        if my_find.standard2 <= 0:
            self.print_info(f"长度估计出错，返回原图...")
            return origin
        
        # 对所有检测框进行判断
        worm_loc = 0  # 幼虫所在位置0无，1左下，2右下，3左下右下都有
        count = 0
        be_catch_count = 0
        for xyxy in boxes:
            cut_image = self.image[
                int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
            ]
            try:
                fast_keypoints = fast_ratio(cut_image, 3)
                circle = fit_circle(fast_keypoints)
                # 为中心下方的两个六边形绘制圆与标签
                is_in = my_find.in_contour(xyxy)
                if is_in:
                    count += 1
                    _, be_catch = draw_circle(
                        self.image,
                        circle,
                        my_find.standard2,
                        (int(xyxy[0]), int(xyxy[1])),
                        thickness = 2,
                        diameterThreshold = self.diameter_threshold
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
                    if be_catch:
                        be_catch_count += 1
                        worm_loc |= is_in
            except Exception as e:
                self.print_info(f"未知错误，返回原图，错误日志如下：\n{e}\n")
                return origin
        self.print_info(f'有{count}只虫，其中有{be_catch_count}只虫需要抓取')
        return self.image, worm_loc

    def catch(self, worm_loc):
        """
        抓幼虫
        
        Arguments:
            worm_loc {int} -- 0(00)无幼虫 1(01)左下有幼虫 2(10)右下有幼虫 3(11)左下右下都有幼虫
        """
        is_ready = self.arm.receive_signal()
        if is_ready and not self.arm.waiting:
            self.print_info(f"输出GPIO信号: {worm_loc:2b}(0无幼虫 1左下有幼虫 10右下有幼虫 11左下右下都有幼虫)")
            self.arm.act(worm_loc)
            self.arm.waiting = True
            t = Thread(target=self.arm.wait_response, args=((self.index, self.config['sleepTime'])))
            t.start()
            self.flag = True
        elif not is_ready:
            if self.flag:
                self.print_info('等待机械臂信号...')
                self.flag = False
                
    def print_info(self, str):
        print(f'--> 摄像头{self.index}：{str}')


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
        self.waiting = False

    def act(self, worm_loc):
        """
        激活机械臂，左下格发送01，右下格发送10，都有发送11，都没有发送00

        Keyword Arguments:
            left {bool} -- 是否是左下格的六边形 (default: {True})
        """
        if worm_loc == 0:
            out0 = False
            out1 = False
        elif worm_loc == 1:
            out0 = False
            out1 = True
        elif worm_loc == 2:
            out0 = True
            out1 = False
        elif worm_loc == 3:
            out0 = True
            out1 = True
        else:
            out0 = False
            out1 = False
        self.gpio_out0.write(out0)
        self.gpio_out1.write(out1)
        

    def receive_signal(self):
        """
        接收机械臂发送的信号

        Returns:
            bool -- True为1，False为0
        """
        return self.gpio_in.read()
    
    def wait_response(self, index, sleepTime):
        time.sleep(sleepTime)
        self.waiting = False

    def close(self):
        """
        释放GPIO
        """
        self.gpio_in.close()
        self.gpio_out.close()
