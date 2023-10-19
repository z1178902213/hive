import cv2
from threading import Thread

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

def show(index, id):
    cap = cv2.VideoCapture(id)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"{index}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

def show_double(camera_list):
    cap_list = []
    for camera_id in camera_list:
        cap_list.append(cv2.VideoCapture(camera_id))
    index = 0
    while True:
        ret, frame = cap_list[index].read()
        if ret:
            frame_640 = cv2.resize(frame, (640, 640))
            cv2.imshow(f"{index}", frame_640)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print(f'{camera_list[index]}不能用')
        index += 1
        if index == len(cap_list):
            index = 0

if __name__ == "__main__":
    camera_list = find_and_check_cameras()
    t_list = []
    print(camera_list)
    show(0, 10)
    # for index, c in enumerate(camera_list):
    #     t = Thread(target=show, args=((index, c)))
    #     t_list.append(t)
    
    # for t in t_list:
    #     print(f'线程运行{t}')
    #     t.start()
    
    # for t in t_list:
    #     t.join()