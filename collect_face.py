import cv2
import argparse
import numpy as np
import tensorflow as tf
import align.detect_face
import os
import sys
import scipy.misc

# 定义边框颜色
color = (0, 255, 0)
# 定义mtcnn的参数
minsize = 30
threshold = [0.6, 0.7, 0.7]
factor = 0.709
# 定义照片储存文件的根目录
path = 'members/dataset/'
# 定义存储检测到的人脸的列表
faces = []

# 定义已知人脸的encodeing

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def catch_ip_camera(ip):
    # 设置获取ip地址
    ip = "rtsp://admin:admin123@" + ip + "//Streaming/Channels/1"
    # 获取摄像头
    cap = cv2.VideoCapture(ip)
    if cap.isOpened():
        print("Connected!")
        return cap
    else:
        print("Fail to connect!")
        return False


def mtcnn_det(frame):
    face_locations, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    return face_locations


def main(args):

    window_name = 'FaceCollect'
    cv2.namedWindow(window_name)
    cap = catch_ip_camera("192.168.1.107")

    if cap:
        print("Connect successfully!")
    else:
        print("Fail to connet.")
        return

    frame_cnt = 0
    face_cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            # 帧计数并获取当前帧信息
            frame_cnt += 1
            frame_size = np.asarray(frame.shape)[0:2]
            # 将当前帧转换为rgb
            rgb_frame = frame[:, :, ::-1]

            # 使用mtcnn实现人脸检测
            face_locations = mtcnn_det(rgb_frame)
            print("I found {} face(s) in frame {}.".format(len(face_locations), frame_cnt))

            for location in face_locations:
                print(location)
                face_cnt += 1
                # mtcnn用法
                det = np.squeeze(location[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-22, 0)
                bb[1] = np.maximum(det[1]-22, 0)
                bb[2] = np.minimum(det[2]+22, frame_size[1])
                bb[3] = np.minimum(det[3]+22, frame_size[0])
                print(bb[0], bb[1], bb[2], bb[3])
                face = frame.copy()[bb[1]: bb[3], bb[0]: bb[2], :]
                faces.append(face)

                # 如果已获得足够人脸，则结束循环
                if face_cnt == args.number:
                    bb[0] = np.maximum(det[0]-100, 0)
                    bb[1] = np.maximum(det[1]-100, 0)
                    bb[2] = np.minimum(det[2]+100, frame_size[1])
                    bb[3] = np.minimum(det[3]+200, frame_size[0])
                    face = frame.copy()[bb[1]: bb[3], bb[0]: bb[2], :]
                    faces.append(face)
                    print("we have already collect {} faces".format(int(args.number)))
                    break
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 4)
                cv2.imshow("face", face)


            # 显示视屏流数据
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
               break
            
            if face_cnt == args.number:
               break

        else:
            print("The connection has broken down!")
            cap.release()
            cap = catch_ip_camera("192.168.1.107")
            if(cap):
                print("The connection has been fixed up.")

    cap.release()
    cv2.destroyAllWindows()

    face_id = len(os.listdir(path))
    if(len(faces) <= 0):
        print("Sorry, There is no face collected.")
    else:
        # 判断是否已经存在对应名字的文件夹
        pic_path = path + "{}/".format(face_id)
        if(not os.path.exists(pic_path)):
            os.mkdir(pic_path)
        # 判断是否识别到规定数量的人脸
        if(len(faces) < args.number):
            print("Sorry, We just collected {} forface(s).".format(len(faces)))
        else:
            print("OK, We sucessfully collected {} face(s)!".format(len(faces)))

        # 写入照片
        for i in range(len(faces) - 1):
            face = cv2.resize(faces[i], (160, 160))
            cv2.imwrite(pic_path+"{}.jpg".format(i), face)
        cv2.imwrite("members/members_pics/{}.jpg".format(face_id), faces[-1])



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('number', type=int,
        help='The number of the faces that be collected', default=60)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
