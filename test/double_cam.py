import cv2
# 通过cv2中的类获取视频流操作对象cap
cap1 = cv2.VideoCapture(12)  # 这里的参数是视频文件的绝对路径
# 调用cv2方法获取cap的视频帧（帧：每秒多少张图片）
fps = cap1.get(cv2.CAP_PROP_FPS)
print(fps)
# 获取cap视频流的每帧大小
size = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)

# 定义编码格式mpge-4
# 一种视频格式，参数搭配固定，不同的编码格式对应不同的参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 定义视频文件输入对象
outVideo = cv2.VideoWriter("test11.mp4", fourcc, fps, size)#第一个参数是保存视频文件的绝对路径
# 获取视频流打开状态
if cap1.isOpened():
    rval, frame = cap1.read()
    print('ture')
else:
    rval = False
    print('False')

tot = 1
c = 1
# 循环使用cv2的read()方法读取视频帧
while rval:
    rval, frame = cap1.read()
    tot += 1
    print('tot=', tot)
    # 使用VideoWriter类中的write(frame)方法，将图像帧写入视频文件
    outVideo.write(frame)
    if tot == 30:
        break
# 释放窗口
cap1.release()
outVideo.release()
cv2.destroyAllWindows()
