import cv2
from threading import Thread


def find_and_check_cameras():
    # 设置cv日志等级，忽视烦人的WARNING和ERROR
    log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)
    ok_list = []
    for i in range(0, 20):  # 遍历摄像头编号，找到能用的双目摄像头
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok_list.append(i)
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
        cap = cv2.VideoCapture(camera_id)
        cap.set(3, 108)
        cap.set(4, 192)
        cap_list.append(cap)
    index = 0
    while True:
        ret, frame = cap_list[index].read()
        if ret:
            # frame_640 = cv2.resize(frame, (640, 640))
            cv2.imshow(f"{index}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print(f"{camera_list[index]}不能用")
        index += 1
        if index == len(cap_list):
            index = 0


def video_save(camera_list):
    outVideo_list = []
    cap_list = []
    for camera_id in camera_list:
        print(f"将{camera_id}加入摄像头列表")
        cap = cv2.VideoCapture(camera_id)
        cap_list.append(cap)
        # get size and fps of video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "2")
        # create VideoWriter for saving
        outVideo = cv2.VideoWriter(
            f"save_test_video_{camera_id}.avi", fourcc, fps, (width, height)
        )
        outVideo_list.append(outVideo)
    index = 0
    while True:
        for cap, outVideo in zip(cap_list, outVideo_list):
            ret, frame = cap.read()
            if ret:
                outVideo.write(frame)
            else:
                print(f"{index}读不了一点")
            index += 1


if __name__ == "__main__":
    camera_list = find_and_check_cameras()
    t_list = []
    print(camera_list)
    show_double(camera_list)
    # for index, c in enumerate(camera_list):
    #     t = Thread(target=show, args=((index, c)))
    #     t_list.append(t)

    # for t in t_list:
    #     print(f'线程运行{t}')
    #     t.start()

    # for t in t_list:
    #     t.join()
