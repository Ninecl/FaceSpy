# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import threading
import math
import os
import math
import cv2
from datetime import datetime as date
import pymysql
import tensorflow as tf
import facenet
import numpy as np
import align.detect_face
import align.detect_face_c
import time
from numba import jit


# 定义颜色
white = (255, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
# 定义mtcnn的参数
minsize = 120
threshold = [0.6, 0.7, 0.7]
factor = 0.709


# facenet计算前需要对照片进行预处理
def pre_process_img(Images):
    pics_ls = []
    for Image in Images:
        processed = cv2.resize(Image, (160, 160))
        prewhitened = facenet.prewhiten(processed)
        pics_ls.append(prewhitened)
    return pics_ls


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

img = cv2.imread("2.jpg")
img = img[:, :, ::-1]
face_locations, face_alignments = align.detect_face.detect_face(img, minsize,
                                                                pnet, rnet, onet,
                                                                threshold, factor)

faces = []
frame_size = np.asarray(img.shape)[0:2]

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
    face = img[bb[1]: bb[3], bb[0]: bb[2], :]
    faces.append(face)
    faces.append(face)

feed_dict = {images_placeholder: pre_process_img(faces), phase_train_placeholder: False}
embs = sess.run(embeddings, feed_dict=feed_dict)
print(faces, len(faces))
print(embs)
print(len(embs))


                


