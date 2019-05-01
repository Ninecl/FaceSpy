# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import multiprocessing
import cv2
import numpy as np
import tensorflow as tf
import align.detect_face
import align.detect_face_c
import face_recognition
from numba import jit
import time


# 定义颜色
white = (255, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
# 定义mtcnn的参数
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
# 跳帧数
frame_jump = 2


def load_mtcnn():
    # 定义tensorflow图，这里包含了mtcnn三个net
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # 加载mtcnn三个net
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def detect(frameQue, que, Mode, e):

    # 定义了显示框，用于显示视频流
    # window_name = 'FaceRecognization'
    # cv2.namedWindow(window_name)

    # 帧计数
    frame_cnt = 0
    # 无人脸帧计数
    no_face_frame_cnt = 0
    # 是否正在记录视频
    if_catching_vedio = False
    # 加载mtcnn
    pnet, rnet, onet = load_mtcnn()

    # 进入读取视频流的循环
    while True:
        ret = frameQue.empty()
        if ret is not True:
            frame = frameQue.get()
            # 帧计数并获取当前帧信息
            frame_cnt += 1
            frame_size = np.asarray(frame.shape)[0:2]
            # 检测程序每隔2帧检测一次人脸
            if frame_cnt % frame_jump == 0:
                # 将当前帧缩小五分之一，并转换为rgb
                small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
                # rgb_frame = frame[:, :, ::-1]
                rgb_small_frame = small_frame[:, :, ::-1]
                t1 = time.time()
                # 使用face_recognition实现人脸检测
                # face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1)

                # 使用mtcnn实现人脸检测
                face_locations, _ = align.detect_face.detect_face(rgb_small_frame, minsize, pnet, rnet, onet, threshold, factor)
                t2 = time.time()
                # print("I found {} face(s) in frame {}.".format(len(face_locations), frame_cnt))

                # 如果检测到人脸，则记录本段视频，直到往后20帧都不出现人脸为止
                if len(face_locations) > 0:
                    que.put(frame.copy())
                    no_face_frame_cnt = 0
                    e.set()
                    if not if_catching_vedio:
                        if_catching_vedio = True
                        print("Process start!")
                else:
                    no_face_frame_cnt += 1
                if if_catching_vedio and no_face_frame_cnt > 20//frame_jump:
                    if_catching_vedio = False
                    que.put(None)
        frame_cnt %= 10000

    # 以下部分当需要使用opencv显示视频流调试时使用
    """
                for location in face_locations:
                    # face_recognition用法"
                    y1, x2, y2, x1 = location
                    x1 *= 4
                    x2 *= 4
                    y1 *= 4
                    y2 *= 4 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # mtcnn用法
                    if location[4] < 0.8:
                        continue
                    # 首先确定缩小5倍情况下的人脸边框
                    det = np.squeeze(location[0:4])
                    # 这里定义两个边框
                    # 一个是在缩小图像中的人脸边框
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-4, 0)
                    bb[1] = np.maximum(det[1]-4, 0)
                    bb[2] = np.minimum(det[2]+4, frame_size[1])
                    bb[3] = np.minimum(det[3]+4, frame_size[0])
                    # 一个是原尺寸中的人脸边框
                    bb_l = np.zeros(4, dtype=np.int32)
                    bb_l[0] = np.maximum(det[0]*5-22, 0)
                    bb_l[1] = np.maximum(det[1]*5-22, 0)
                    bb_l[2] = np.minimum(det[2]*5+22, frame_size[1])
                    bb_l[3] = np.minimum(det[3]*5+22, frame_size[0])
                    # 框出这张人脸图片
                    # face = frame[bb_l[1]: bb_l[3], bb_l[0]: bb_l[2], :]
                    # 标出人脸框
                    cv2.rectangle(frame, (bb_l[0], bb_l[1]), (bb_l[2], bb_l[3]), green, 2)
            # 绘制竖线标
            for i in range(1, 10):
                x = i * 128
                cv2.line(frame, (x, 0), (x, 720), green, 2)
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(1)
            if(c & 0xFF == ord('q')):
               break
    # cv2.destroyAllWindows()
    """
