# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import threading
import math
import os
import cv2
from datetime import datetime as date
import pymysql
import tensorflow as tf
import facenet
import numpy as np
import align.detect_face
import time


# 定义颜色
white = (255, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
# 定义mtcnn的参数
minsize = 120
threshold = [0.6, 0.7, 0.7]
factor = 0.709
# 定义opencv保存视频参数
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# 定义fps
fps = 25
# 定义出现在摄像头中的所有人脸
faces_in_camera = []
# 定义一存放所有face的img，pos，flags
face_emb_pos_all = []
# 定义一个存放线程的变量
threads = []
# 定义一个线程锁，保证之后的线程同步
lock = threading.RLock()


class A_face(object):
    '为检测到的人脸定义一个类，方便记录相关数据'

    def __init__(self, img, emb, time, front_flag, pos):
        self.id = -1
        self.no_match_cnt = 0
        self.embs = [emb, ]
        self.time = time
        self.face_ls = [img, ]
        self.front_flags = [front_flag, ]
        self.path = [pos, ]
        self.n_x, self.n_y = pos

    def predict_pos(self, pos):
        '根据加速度计算下一次的位置'
        if len(self.path) >= 3:
            # print("dx + dy: {}".format(math.sqrt((self.n_x - pos[0]) ** 2 + (self.n_y - pos[1]) ** 2)))
            x1, y1 = self.path[-3]
            x2, y2 = self.path[-2]
            x3, y3 = self.path[-1]
            v0_x = x2 - x1
            v0_y = y2 - y1
            v1_x = x3 - x2
            v1_y = y3 - y2
            a_x = v1_x - v0_x
            a_y = v1_y - v0_y
            dx = v1_x + 1/2 * a_x
            dy = v1_y + 1/2 * a_y
            self.n_x = x3 + dx
            self.n_y = y3 + dy


def check(face, mode, pos, cnt):
    check = False
    dir_path = "pics_history/"

    if mode == "IN":
        dir_path += "IN/"
        if pos == "RIGHT":
            dir_path += "Right/" + face.time + '_' + str(cnt)
            if face.path[0][0] >= 900 and face.path[-1][0] <= 800:
                check = True
        elif pos == "MIDDLE":
            dir_path += "Middle/" + face.time + '_' + str(cnt)
            if face.path[0][1] <= 200 and face.path[-1][1] >= 500:
                check = True
        elif pos == "LEFT":
            dir_path += "Left/" + face.time + '_' + str(cnt)
            if face.path[0][0] <= 200 and face.path[-1][0] >= 400:
                check = True
    else:
        dir_path += "OUT/"
        if pos == "LEFT":
            dir_path += "Left/" + face.time + '_' + str(cnt)
            if face.path[0][0] <= 300 and face.path[-1][0] >= 500:
                check = True
        elif pos == "MIDDLE":
            dir_path += "Middle/" + face.time + '_' + str(cnt)
            if face.path[0][1] <= 300 and face.path[-1][1] >= 550:
                check = True
        elif pos == "RIGHT":
            dir_path += "Right/" + face.time + '_' + str(cnt)
            if face.path[0][0] >= 1000 and face.path[-1][0] <= 800:
                check = True



    isExists = os.path.exists(dir_path)
    if not isExists:
        os.makedirs(dir_path)
    for i in range(len(face.face_ls)):
        if face.front_flags[i]:
            cv2.imwrite(dir_path + "/" + str(i) + ".jpg", face.face_ls[i][:, :, ::-1])
        else:
            cv2.imwrite(dir_path + "/" + str(i) + "_not_front.jpg", face.face_ls[i][:, :, ::-1])

    return check


# 模拟操作系统cpu预测算法，在alp越大，最近的一张照片越重要，历史照片越不重要
def alpha(ls, alp):
    if len(ls) <= 1:
        return ls[0]
    else:
        res = ls[0]
        for i in ls[1:]:
            res = i*alp + res*(1-alp)
        return res


# 封装alp算法，对一系列人脸照片中的近10张计算alp值
def calculate_alpha(emb, ls):
    for i in range(len(ls)):
        dis_ls = []
        compare_len = min(len(ls), 10)
        for e in ls[-compare_len:]:
            dis_ls.append(np.linalg.norm(emb - e))
        dis = alpha(dis_ls, 0.6)
    return dis


# facenet计算前需要对照片进行预处理
def pre_process_img(Images):
    pics_ls = []
    for Image in Images:
        processed = cv2.resize(Image, (160, 160))
        prewhitened = facenet.prewhiten(processed)
        pics_ls.append(prewhitened)
    return pics_ls


# 计算一张人脸的emb
def caculate_one_face_emb(image, images_placeholder, embeddings, phase_train_placeholder, sess):
    feed_dict = {images_placeholder: pre_process_img(image), phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)
    emb = embs[0]
    return emb


def load_graph():
    # 定义facenet的tensorflow图和会话
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            # 加载MTCNN三个网络
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            # 加载facenet模型
            facenet.load_model("models/20180402-114759.pb")
            # 设置facenet计算张量
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    return sess, pnet, rnet, onet, images_placeholder, embeddings, phase_train_placeholder


def process_faces(face_emb_pos_all, faces,
                     images_placeholder, embeddings, phase_train_placeholder,
                     sess, positions, front_flags, nowTime, mTime):
    """
    该函数用于在处理每一帧过程中，对每一帧中的一张人脸进行emb计算和match操作，对每一张人脸分开操作是为了开启多线程
    """
    # 计算这张人脸的emb
    face_emb_pos = [mTime, ]
    feed_dict = {images_placeholder: pre_process_img(faces), phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)
    for i in range(len(embs)):
        face_emb_pos.append([faces[i], embs[i], nowTime, front_flags[i], positions[i]])
    lock.acquire()
    idx = len(face_emb_pos_all)
    while(idx - 1 > 0):
        if face_emb_pos[0] >= face_emb_pos_all[idx - 1][0]:
            face_emb_pos_all.insert(idx, face_emb_pos)
            break
        else:
            idx -= 1
    else:
        face_emb_pos_all.insert(idx, face_emb_pos)
    lock.release()


def match_faces(faces_emb_pos):
    # print("A new alpha_ls: ({}, {})".format(len(faces_emb_pos), len(faces_in_camera)))
    alp_ls = np.zeros((len(faces_emb_pos), len(faces_in_camera)))
    vis = np.zeros(len(faces_emb_pos))
    for i in range(len(faces_emb_pos)):
        emb = faces_emb_pos[i][1]
        for j in range(len(faces_in_camera)):
            alp = calculate_alpha(emb, faces_in_camera[j].embs)
            alp_ls[i][j] = alp

    # 输出alpha值列表
    for i in range(len(faces_emb_pos)):
        for j in range(len(faces_in_camera)):
            print("%.4f" % alp_ls[i][j], end=" ")
        print()

    # 这里一共有两种情况
    # 第一种情况是检测到的人脸数比已记录的人脸数多，第二种是检测到的人脸数小于等于已检测到的人脸数
    # 首先简单分析一下匹配方案
    # 对于n张人脸，与已记录的m张人脸计算近10张照片的alp距离，形成一个n*m的矩阵
    # 在这个矩阵中，对于cnt个人脸，每次匹配矩阵中的alpdis最小的那张人脸
    # 并在这次匹配完成后将矩阵中对应的行和列置为10，方便下一张人脸的匹配
    cnt = min(len(faces_in_camera), len(faces_emb_pos))
    for i in range(cnt):
        match_f = False
        minnum = np.min(alp_ls)
        p = np.where(alp_ls == minnum)
        x = p[0][0]
        y = p[1][0]

        # 结合计算出的欧式距离和位置预判判断是否是跟踪对象
        if len(faces_in_camera[y].path) < 3:
            if minnum < 0.85:
                match_f = True
            else:
                continue
        else:
            x1, y1 = faces_in_camera[y].n_x, faces_in_camera[y].n_y
            x2, y2 = faces_emb_pos[x][4]
            dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            print("dx + dy: {}".format(dis))
            if minnum <= 0.8 or dis <= 10:
                match_f = True
            elif minnum <= 1.00 and dis <= 100:
                match_f = True

        if match_f:
            vis[x] = 1
            alp_ls[x] = 10
            alp_ls[:, y:y+1] = 10
            faces_in_camera[y].face_ls.append(faces_emb_pos[x][0])
            faces_in_camera[y].embs.append(faces_emb_pos[x][1])
            faces_in_camera[y].front_flags.append(faces_emb_pos[x][3])
            faces_in_camera[y].path.append(faces_emb_pos[x][4])
            if len(faces_in_camera[y].path) < 3:
                faces_in_camera[y].n_x, faces_in_camera[y].n_y = faces_emb_pos[x][4]
        else:
            continue

    # 如果检测到的人脸数比已有的人脸数多，那么很显然n*m矩阵不会全被置10
    # vis列表标志了哪些人脸已被匹配，未被匹配的人脸则创建新的人脸对象
    for i in range(len(faces_emb_pos)):
        if vis[i] == 0:
            new_face = A_face(faces_emb_pos[i][0], faces_emb_pos[i][1],
                              faces_emb_pos[i][2], faces_emb_pos[i][3],
                              faces_emb_pos[i][4])
            faces_in_camera.append(new_face)


def now_time():
    now = date.now()
    year = str(now.year)
    month = str('%02d' % now.month)
    day = str('%02d' % now.day)
    hour = str('%02d' % now.hour)
    minute = str('%02d' % now.minute)
    second = str('%02d' % now.second)
    return year+month+day+hour+minute+second


def process(videoQue, alreadyQue, Mode, Pos, e):

    # 加载MTCNN与facenet
    sess, pnet, rnet, onet, images_placeholder, embeddings, phase_train_placeholder = load_graph()
    # 定义显示窗口
    # window_name = 'FaceProcess'
    # cv2.namedWindow(window_name)
    # 定义一个变量， 来记录每天的游客id
    id_cnt = 0

    while True:
        # 获取当前时间，年月日时分秒
        nowTime = now_time()
        # 获取当前时间，毫秒
        mTime = time.time()
        # 设置阻塞，如果队列为空的话就设置阻塞
        e.wait()
        frame = None
        f = videoQue.empty()
        if f:
            e.clear()
        else:
            frame = videoQue.get()
            if frame is not None:
                """
                # 写视频
                intime = time.ctime()
                outname = intime + '.avi'
                outfdangfar(outname, fourcc, fps, (1280, 720))
                """
                # 获取每一帧的数据信息
                rgb_frame = frame[:, :, ::-1]
                frame_size = np.asarray(frame.shape)[0:2]
                t1 = time.time()
                face_locations, face_alignments = align.detect_face.detect_face(rgb_frame, minsize,
                                                                                pnet, rnet, onet,
                                                                                threshold, factor)
                t2 = time.time()
                # 这一帧中所有的front_flag
                front_flags = []
                # 这一帧中所有的pos
                positions = []
                # 这一帧中出现的所有人脸
                faces = []
                for i in range(len(face_locations)):
                    face_location = face_locations[i]
                    if face_location[4] < 0.8:
                        continue
                    det = np.squeeze(face_location[0: 4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-22, 0)
                    bb[1] = np.maximum(det[1]-22, 0)
                    bb[2] = np.minimum(det[2]+22, frame_size[1])
                    bb[3] = np.minimum(det[3]+22, frame_size[0])
                    # cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), green, 5)
                    # 框出人脸
                    face = rgb_frame[bb[1]: bb[3], bb[0]: bb[2], :]
                    face_width = bb[2] - bb[0]
                    face_highth = bb[3] - bb[1]
                    # 标出这张人脸的中间线
                    x_mid_line = (bb[0] + bb[2]) / 2
                    # 标出眼睛，鼻子，嘴角的x,y
                    left_eye_x, left_eye_y = face_alignments[0][i], face_alignments[5][i]
                    right_eye_x, right_eye_y = face_alignments[1][i], face_alignments[6][i]
                    nose_x, nose_y = face_alignments[2][i], face_alignments[7][i]
                    left_mouth_x, left_mouth_y = face_alignments[3][i], face_alignments[8][i]
                    right_mouth_x, right_mouth_y = face_alignments[4][i], face_alignments[9][i]
                    # 计算人脸在整个视频中的位置(人脸的中心点)
                    pos = ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)
                    # 比对人脸鼻子，眼睛，嘴在整个人脸框中的位置，大致判断人脸是否为正脸
                    # 判断鼻子是否在人脸框的较中间位置
                    """
                    注意！这里需要更改！
                    """
                    front_flag = False
                    nose_flag = abs(nose_x - x_mid_line) < (face_width / 6)
                    eye_flag = math.sqrt((right_eye_x - left_eye_x) ** 2 + (left_eye_y - right_eye_y) ** 2) > (face_width / 5)
                    if nose_flag and eye_flag:
                        front_flag = True
                    faces.append(face)
                    positions.append(pos)
                    front_flags.append(front_flag)
                if len(faces) <= 0:
                    continue
                # 对每一个人脸启动一个线程开始计算
                t = threading.Thread(target=process_faces, args=(face_emb_pos_all, faces,
                                                                    images_placeholder,
                                                                    embeddings,
                                                                    phase_train_placeholder,
                                                                    sess, positions, front_flags, nowTime, mTime))
                threads.append(t)
                t.start()

                """
                cv2.imshow(window_name, frame)
                c = cv2.waitKey(1)
                if(c & 0xFF == ord('q')):
                    break
                """
            else:
                """
                if Mode == "IN":
                    print("IN: ", faces_in_camera)
                else:
                    print("OUT: ", faces_in_camera)
                """
                for t in threads:
                    t.join()
                threads.clear()
                for face_emb_pos in face_emb_pos_all:
                    face_emb_pos = face_emb_pos[1: ]
                    match_faces(face_emb_pos)
                    for face in faces_in_camera:
                        face.predict_pos(face_emb_pos[0][4])
                face_emb_pos_all.clear()
                # 在这一段视频分析结束后，分析faces_in_camera列表中的所有人脸，如果是分析位置判断出人脸是通过的，那么则记录该人脸信息
                cnt = 0
                # put_face = False
                for face in faces_in_camera:
                    cnt += 1
                    if len(face.face_ls) <= 4:
                        continue
                    # put_face = True
                    if Pos == "RIGHT":
                        print("right:", face.path)
                    elif Pos == "MIDDLE":
                        print("middle:", face.path)
                    else:
                        print("left:", face.path)
                    # 下面的emb剪切要根据具体情况考虑，emb过少时直接省略
                    # length = len(face.embs)
                    # face.face_ls = face.face_ls[math.ceil(length*0.14): int(length*0.43)]
                    # face.embs = face.embs[math.ceil(length*0.14): int(length*0.43)]
                    face.face_ls = face.face_ls[: -3]
                    face.embs = face.embs[: -3]
                    face.front_flags = face.front_flags[: -3]
                    # 检查人脸是否进入
                    if check(face, Mode, Pos, cnt):
                        # 将每一个人脸加入到人脸列表中，标识为已进入博物馆
                        alreadyQue.put([face, True])
                    else:
                        alreadyQue.put([face, False])

                # if put_face:
                if Pos == "RIGHT":
                    alreadyQue.put("rightNone")
                elif Pos == "MIDDLE":
                    alreadyQue.put("middleNone")
                elif Pos == "LEFT":
                    alreadyQue.put("leftNone")
                faces_in_camera.clear()
