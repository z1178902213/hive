# hive

蜂巢识别项目

## TODO
- [x] 在中心图片中心加入一个“+”形状的准星
- [x] 优化边缘检测算法，能够更加准确匹配幼虫的外接圆
- [x] YOLO5检测精度优化，能够识别身边没有蜂蜡的蜜蜂幼虫
- [x] 修复外接圆绘制不完整的问题
- [x] 动态调整输出图片的文字标签字体大小
- [x] 转换RKNN并成功在RK3399pro上进行检测
- [x] 优化程序逻辑，读取一帧图片识别一张图片
- [x] 测试一帧图像处理平均速率
- [x] 六边形检测优化，准确识别中点下方两个六边形方框
- [x] 使用python进行GPIO调用
- [ ] 使用双路摄像头提高检测精度

## 使用方式
### 1 安装依赖文件
```shell
pip3 install -r requirements.txt
```

### 2 修改参数
打开`main.py`文件，可以看到其中有一些大写命名的参数，请根据需求进行修改：  
```python
RKNN_MODEL = "worm.rknn"
BOX_THRESH = 0.5
NMS_THRESH = 0.0
IMG_SIZE = 640
RESHAPE_RATIO = 3  # 在进行角点检测的时候所进行的放大比例
IMAGE_FOLDER = "./all_images"
OUTPUTS_ROOT = IMAGE_FOLDER + "_outputs"
PROBLEM_ROOT = "./problems"
CLASSES = "worm"
```

### 3 运行项目主程序文件
```shell
python3 main.py
```
