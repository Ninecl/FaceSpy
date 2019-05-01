# -*- coding: utf-8 -*-
import time
import pickle
from datetime import datetime as date
import numpy as np
import pymysql
import cv2
import tensorflow as tf
import facenet


def load_graph():
    # 定义facenet的tensorflow图和会话
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            # 加载facenet模型
            facenet.load_model("models/20180402-114759.pb")
            # 设置facenet计算张量
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    return sess, images_placeholder, embeddings, phase_train_placeholder


def now_time():
    now = date.now()
    year = str(now.year)
    month = str('%02d' % now.month)
    day = str('%02d' % now.day)
    hour = str('%02d' % now.hour)
    minute = str('%02d' % now.minute)
    second = str('%02d' % now.second)
    return year+month+day+hour+minute+second


def front_face_ls(face):
    """
    过滤人脸的embs数组，取出所有标记为正脸的人脸的emb
    """
    return np.asarray([face.embs[i] for i in range(len(face.embs)) if face.front_flags[i]])

def match_face(face, pass_flag, ls):
    # 计算这一张人脸的每一个emb与人脸列表中每一个人脸的每一个emb的dis并计算均值
    faceEmbs = front_face_ls(face)
    if len(faceEmbs) <= 0:
        return
    ls_dis = []
    for i in range(len(ls)):
        ls_face = ls[i][0]
        embs = front_face_ls(ls_face)
        if len(embs) <= 0:
            ls_dis.append(10)
            continue
        sum_dis = 0
        cnt = 0
        for emb in faceEmbs:
            sum_dis += np.average(np.linalg.norm(embs-emb, axis=1))
            cnt += 1
        else:
            ls_dis.append(sum_dis / cnt)
    # 在0.905的阈值下寻找最小的dis，如果可以找到，则添加对应的位置星系
    # 如果找不到，就创建一个新的人脸
    print("ls_dis: ", ls_dis)
    if len(ls_dis) <= 0:
        ls.append([face, pass_flag, 1])
    else:
        idx = np.where(ls_dis == np.min(ls_dis))[0][0]
        min_dis = ls_dis[idx]
        if min_dis <= 0.905:
            ls[idx][0].face_ls += face.face_ls
            ls[idx][0].embs += face.embs
            ls[idx][0].front_flags += face.front_flags
            if pass_flag:
                ls[idx][1] = True
            ls[idx][2] += 1
        else:
            ls.append([face, pass_flag, 1])

def collect_cnt_person(alreadyQue, Mode):
    # 加载数据库配置文件
    with open("./dbset.pkl", 'rb') as infile:
        dbSet = pickle.load(infile)
    # 连接数据库，当识别出某人离开时，对数据库中的记录进行操作
    conn = pymysql.connect("39.98.90.118", "zyf", "zyf123456", "face_recognition", charset="utf8")
    cursor = conn.cursor()

    # 加载MTCNN图和分类文件
    with open("models/20190423_knn_16people.pkl", "rb") as infile:
        model = pickle.load(infile)
        embs = model["embs"]
        labels = model["labels"]
        classes_num = model["classes"]

    # 创建两个列表，从共享队列中读出的人脸数据经过match后均存放在这两个列表中
    allPassList = []
    # 最终记录有哪些人在内的列表
    person_pass_ls = []
    # 记录从进入共享队列中读到的三个摄像头的none数
    leftNoneCnt = 0
    rightNoneCnt = 0
    middleNoneCnt = 0
    # 定义一个变量，来记录每天的游客id
    id_cnt = 0
    # 定义两个时间来判断是否是新的一天
    lastTime = now_time()
    nowTime = now_time()
    # 记录读取列表时空的次数
    emptyCnt = 0

    # 无限循环读列表（注意，这个操作每两秒一次，最后设置了阻塞2s）
    while True:
        # 首先判断是否是新的一天, 如果是, 则id_cnt重置
        lastTime = nowTime
        nowTime = now_time()
        if lastTime[6: 8] != nowTime[6: 8]:
            id_cnt = 0

        emptyf = alreadyQue.empty()
        if emptyf:
            emptyCnt += 1
        else:
            emptyCnt = 0

        while not emptyf and emptyCnt < 2:
            face = alreadyQue.get()
            if face == "rightNone":
                rightNoneCnt += 1
            elif face == "leftNone":
                leftNoneCnt += 1
            elif face == "middleNone":
                middleNoneCnt += 1
            else:
                the_face = face[0]
                pass_flag = face[1]
                match_face(the_face, pass_flag, allPassList)
            emptyf = alreadyQue.empty()



        print(rightNoneCnt, middleNoneCnt, leftNoneCnt)
        if rightNoneCnt + middleNoneCnt + leftNoneCnt >= 2 and emptyCnt > 1:
            if Mode == "IN":
                print("IN:", allPassList)
            else:
                print("OUT", allPassList)
            for face_record in allPassList:
                if face_record[2] < 2 or not face_record[1]:
                    continue
                id_cnt += 1
                face = face_record[0]
                # 检查这张人脸是否是成员
                pro_ls = np.zeros(classes_num)
                member_idx = -1
                for emb in face.embs:
                    dis_ls = np.linalg.norm(embs-emb, axis=1)
                    print(dis_ls)
                    min_dis = np.min(dis_ls)
                    if min_dis <= 0.80:
                        min_idx = np.where(dis_ls == min_dis)[0][0]
                        pro_ls[labels[min_idx]] += 1
                idx = np.where(pro_ls == np.max(pro_ls))[0][0]
                print(pro_ls)
                # print(pro_ls)
                # print(len(predictions))
                if pro_ls[idx] >= len(face.embs) // 2:
                    member_idx = idx
                    if Mode == "IN":
                        print("Member {} in.".format(member_idx))
                    else:
                        print("Member {} out.".format(member_idx))
                # 将这条数据放入person_in_ls
                # face_record.append(member_idx)
                # person_in_ls.append(face_record)

                # 通过id_cnt合成这张人脸的id
                face.id = face.time + str(id_cnt)
                # 向数据库中写入这张人脸
                if Mode == "IN":
                    sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                else:
                    sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                # 生成存入数据库内的各条数据信息
                # 图片
                img = face.face_ls[len(face.face_ls)//2][:, :, ::-1]
                img = cv2.imencode('.jpg', img)[1]
                img = np.array(img).tostring()
                # 名字(日期+cnt, 用来在网站上显示)
                Name = "Vistor"
                if member_idx >= 0:
                    cursor.execute('select Name from members where ID = %s' % str(member_idx))
                    data = cursor.fetchone()
                    Name = "Member_" + str(member_idx)
                # 日期
                Date = face.time[0: 8]
                # 时间
                passTime = face.time[8: 14]
                # 执行数据库操作
                cursor.execute(sql, (face.id, passTime, pymysql.Binary(img), Name, Date, str(member_idx)))
                if member_idx > -1:
                    if Mode == "IN":
                        sql = "update members set inside = 1 where ID = {}".format(member_idx)
                    else:
                        sql = "update members set inside = 0 where ID = {}".format(member_idx)
                    cursor.execute(sql)
                conn.commit()
            allPassList.clear()
            leftNoneCnt = 0
            rightNoneCnt = 0
            middleNoneCnt = 0


        """
        # 接下来判断离开的人脸列表中是否有与进入的人脸列表中匹配的人脸
        # 我们对于每个离开的人脸，和进入的人脸一一比对，离开的人脸有15张，进入的人脸也有15张，随机比对100次
        # 如果有90张以上比较值小于0.95，则判断为同一个人（注意，facenet官方给定小于1.05即为同一张人脸，我们再缩小到0.95，可后期调整）
        # 判断为同一人后，对数据库进行操作，删除列表中这个进入和离开的人脸
        # 因此每次只要读取进入的人脸列表的长度，既可知道有多少人在内
        i = 0
        while(i < len(alreadyOutList)):
            cnt_true = 0
            cnt_false = 0
            j = 0
            if_find = False
            sum_dis = 0
            cnt = 0
            while(j < len(alreadyInList) and not if_find):
                in_embs = alreadyInList[j].embs
                out_embs = alreadyOutList[i].embs
                rans = random.sample(range(0, len(in_embs)), min(10, len(in_embs)))
                for out_emb in out_embs:
                    for ran in rans:
                        dis = np.linalg.norm(out_emb - in_embs[ran])
                        sum_dis += dis
                        cnt += 1
                        if dis <= 0.95:
                            cnt_true += 1
                        else:
                            cnt_false += 1
                print("average: ", sum_dis / cnt)
                if(sum_dis / cnt < 0.85 and int(alreadyOutList[i].time) > int(alreadyInList[j].time)):
                    if_find = True
                    sql = "UPDATE faces set DepartureTime = %s, Inside = '0' where ID = %s"
                    cursor.execute(sql, (alreadyOutList[i].time[8: 14], alreadyInList[j].id))
                    conn.commit()
                    del alreadyOutList[i]
                    del alreadyInList[j]
                else:
                    if_find = False
                j += 1
            if if_find:
                continue
            else:
                alreadyOutList[i].no_match_cnt += 1
                i += 1
                print("next!")

        # 对于alreadyOutList中的人脸，如果匹配20次都无法匹配，则删除
        i = 0
        while(i < len(alreadyOutList)):
            if alreadyOutList[i].no_match_cnt >= 20:
                del alreadyOutList[i]
            else:
                i += 1
        """
        if Mode == "IN":
            print("There are {} person(s) have came in.".format(id_cnt))
        else:
            print("There are {} person(s) have came out.".format(id_cnt))
        # print(person_in_ls)
        time.sleep(2)
