import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label


class FindContour(object):
    def __init__(self, image: np.array, topk: int, draw_contour: bool = False, draw_circle: bool = False,
                 is_draw_doji: bool = True, doji_len: int = 10):
        """
        :param image:  open_cv读取的图像，np.array
        :param topk:   需要找到的下半部分中离中点最近的框的个数
        :param draw_contour:   是否在图像上画出边框
        :param draw_circle:   是否在图像上画出内接圆
        """
        self.image = image
        h, w, _ = image.shape
        self.h, self.w = h, w
        # self.mid_point = np.array([h // 2, w // 2])
        self.center_point = np.array([w // 2, h // 2])
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(image, 172, 255, cv2.THRESH_BINARY)
        self.binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        self.is_draw_doji = is_draw_doji
        self.draw_doji_len = doji_len
        self.topk_cont = self.find_contours(topk, draw_contour)
        self.draw_doji(self.center_point, self.draw_doji_len, (255, 0, 0))
        if len(self.topk_cont) >= 2:
            self.standard1, self.standard2 = self.calculate_standard(
                self.topk_cont, draw_circle)
        else:
            self.standard1, self.standard2 = 0, 0

    def find_contours(self, topk, draw=False):
        threshold_binary = np.where(self.gray > 200, 1, 0)
        self.threshold_binary = threshold_binary * 255
        self.adaptive_threshold = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY_INV,
                                                        11, 2)
        # 进行腐蚀膨胀操作
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(self.threshold_binary.astype(np.uint8), kernel, iterations=1)
        threshold_binary = cv2.erode(self.threshold_binary.astype(np.uint8), kernel, iterations=1)

        label_images, nums_image = label(threshold_binary)
        sizes = np.bincount(label_images.ravel())
        sizes[0] = 0
        max_label = sizes.argmax()
        mask = np.where(label_images == max_label, 0, 1)
        contours, _ = cv2.findContours(mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if
                    cv2.contourArea(cnt) > 2000]  # and not np.all(cnt[..., 1] <= 1.08 * self.h / 2)]
        mid_dis = self.calculate_dis(contours, self.center_point)
        mid_dis.sort(key=lambda x: x[1])
        if len(mid_dis) == 0:
            return mid_dis
        center_contour = contours[mid_dis[0][0]]
        M = cv2.moments(center_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if self.is_draw_doji:
            self.draw_doji((cX, cY), self.draw_doji_len)
        contours = [cnt for cnt in contours if
                    not np.any(np.abs(cnt[..., 1] - cY) < 20) and not np.all(cnt[..., 1] <= cY)]
        result = self.calculate_dis(contours, (cX, cY))
        result.sort(key=lambda x: x[1])
        find_topk = []
        if len(result) < topk:
            return []
        for i in range(topk):
            find_topk.append(contours[result[i][0]])
        find_topk.sort(key=lambda x: np.min(x[0][..., 0]))
        if draw:
            for cnt in find_topk:
                cv2.drawContours(self.image, cnt, -1, (0, 255, 0), 2)
        return find_topk

    def calculate_dis(self, contours, point, X=1, Y=1):
        pX, pY = point
        result = []
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            distance = np.sqrt(X * (cX - pX) ** 2 + Y * (cY - pY) ** 2)
            result.append((i, distance))
        return result

    def inscribed_circle(self, cont: np.array):
        M = cv2.moments(cont)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        min_dist = np.average(
            np.sqrt((cont[:, 0, 0] - cX) ** 2 + (cont[:, 0, 1] - cY) ** 2))
        center = (cX, cY)
        return min_dist, center

    def calculate_standard(self, contours, draw=False):
        circles = []
        for cont in contours:
            max_radius, center = self.inscribed_circle(cont)
            if draw:
                cv2.circle(self.image, center, int(max_radius), (0, 255, 0), 2)
            circles.append((max_radius, np.array(center)))

        standard1 = circles[0][0] + circles[1][0] / 2
        standard2 = np.sqrt(np.sum((circles[0][1] - circles[1][1]) ** 2))
        return standard1 / 5, standard2 / 5.3

    def in_contour(self, rect):
        """
        :param rect: 判断是否在这张图片中的候选框[x,y,x,y]，表示左上角的点和右下角的点
        :return: 返回一个bool值表示候选框是否在图像的topk个下半部分蜂巢的位置中
        :Note: 通过判断候选框的四个角点是否都在在轮廓中，可能会出现误差
        """
        # rect_points = [
        #     (rect[0], rect[1]),  # 左上
        #     (rect[2], rect[1]),  # 右上
        #     (rect[2], rect[3]),  # 右下
        #     (rect[0], rect[3])  # 左下
        # ]
        rect_points = [
            ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)  # 中心点
        ]
        result = 0
        for i, cont in enumerate(self.topk_cont):
            all_points_inside = all(cv2.pointPolygonTest(
                cont, pt, False) >= 0 for pt in rect_points)
            if all_points_inside:
                result = i+1
        return result

    def draw_doji(self, center_point, length=50, color=(0, 0, 255)):
        cx, cy = center_point
        cv2.line(self.image, (cx - length, cy), (cx + length, cy), color, 2)
        cv2.line(self.image, (cx, cy - length), (cx, cy + length), color, 2)


if __name__ == '__main__':
    for img in [img for img in os.listdir('../imgs') if img.endswith('.jpg')]:
        img_name = os.path.splitext(img)[0]
        image = cv2.imread('../imgs/{}'.format(img), cv2.THRESH_BINARY_INV)
        findcontours = FindContour(image, 2, False, True)
        plt.imshow(cv2.cvtColor(findcontours.image, cv2.COLOR_BGR2RGB))
        # plt.imshow(findcontours.dilated)
        plt.show()
        print(" ")
