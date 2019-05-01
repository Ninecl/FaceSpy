# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2


def catch_ip_camera(ip):
    # 设置获取ip地址pInProcess = Process(target=videoProcess.Process_IN, args=(vedioInQueue, alreadyInList, ))
    rtsp = "rtsp://admin:admin123@" + ip + "//Streaming/Channels/1"
    # 获取摄像头
    cap = cv2.VideoCapture(rtsp)

    if cap.isOpened():
        print(ip + ": Connected!")
        return cap
    else:
        print("Fail to connect!")
        return False


def recieve(que, ip):

    # 定义了显示框，用于显示视频流
    # window_name = 'FaceRecognization'
    # cv2.namedWindow(window_name)
    cap = catch_ip_camera(ip)

    if not cap:
        return
    # 进入读取视频流的循环
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            que.put(frame)
            """
            # 绘制竖线标
            for i in range(1, 10):
                x = i * 128
                cv2.line(frame, (x, 0), (x, 720), green, 2)
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(1)
            if(c & 0xFF == ord('q')):
               break
            """
        else:
            print("The connection has broken down!")
            cap.release()
            cap = catch_ip_camera(ip)
            if cap:
                print(ip + ": The connection has been fixed up.")
        # print("One frame use {} and face locate use {}".format(t2-t1, t4-t3))

    cap.release()
    cv2.destroyAllWindows()
