# -*- coding: utf-8 -*-:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import align.detect_face
import align.detect_face_c
import time
import argparse
import sys


# 定义颜色
white = (255, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
# 定义mtcnn的参数
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        # 加载MTCNN三个网络
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)



def catch_ip_camera(ip):
    # 设置获取ip地址
    rtsp = "rtsp://admin:admin123@" + ip + "//Streaming/Channels/1"
    # 获取摄像头
    cap = cv2.VideoCapture(rtsp)

    if cap.isOpened():
        print(ip + ": Connected!")
        return cap
    else:
        print("Fail to connect!")
        return False


def main(args):

    # 定义了显示框，用于显示视频流
    window_name = 'FaceRecognization'
    cv2.namedWindow(window_name)
    cap = catch_ip_camera(args.ip)
    # 定义video
    video = []
    detect_cnt = 0

    if not cap:
        return
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            rgb_frame = frame[:, :, ::-1]
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.2, fy=0.2)
            frame_size = np.asarray(frame.shape)[0:2]
            face_locations, _ = align.detect_face.detect_face(small_frame, minsize,
                                                  pnet, rnet, onet, threshold, factor)
            for face_location in face_locations:
                detect_cnt += 1
                det = np.squeeze(face_location[0:4])
                det *= 5
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-22, 0)
                bb[1] = np.maximum(det[1]-22, 0)
                bb[2] = np.minimum(det[2]+22, frame_size[1])
                bb[3] = np.minimum(det[3]+22, frame_size[0])
                left = bb[0]
                top = bb[1]
                right = bb[2]
                bottom = bb[3]
                if detect_cnt >= 0:
                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), green, 3)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), red, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, "ZhangYufeng", (left + 6, bottom - 6), font, 1.0, white, 1)

            video.append(frame)
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
               break
        else:
            print("The connection has broken down!")
            cap.release()
            cap = catch_ip_camera(str(args.ip))
            if cap:
                print(str(args.ip) + ": The connection has been fixed up.")
        # print("One frame use {} and face locate use {}".format(t2-t1, t4-t3))

    fps = 25
    size = (1280, 720)
    writer = cv2.VideoWriter("ours.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for frame in video:
        writer.write(frame)

    cap.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('ip', type=str,
            help='camera ip')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
