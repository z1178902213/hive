from detect import parse_opt
from detect import run as yolo_detect
import cv2

WEIGHTS = '.\\worm.pt'
SOURCE = '.\\test.png'
IMGSZ = (640, 640)
DATA = '.\\bee_children_v1\\data.yaml'

if __name__ == '__main__':
    opt = parse_opt()
    opt.weights = WEIGHTS
    opt.source = SOURCE
    opt.imgsz = IMGSZ
    opt.data = DATA
    # img = cv2.imread(SOURCE)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # print(vars(opt))
    yolo_detect(**vars(opt))
