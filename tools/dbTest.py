import pymysql
import numpy as np
import cv2


names = ["Yufeng Zhang", "Jiajun Chen", "Runze Jiao", "Yaowen Fan"]
conn = pymysql.connect('localhost', 'root', 'admin123', 'face_recognition')
cursor = conn.cursor()

id_cnt = 0
for i in range(0, 31):
    date = "201905" + "%02d" % i
    for j in range(0, 8):
        time = "%02d" % j
        cnt = np.random.randint(0, 3)
        for k in range(0, cnt):
            time += "%02d" % np.random.randint(0, 60)
            time += "%02d" % np.random.randint(0, 60)
            v_m = np.random.randint(0, 2)
            if v_m == 0:
                name = 'Vistor'
                ID = date + time + '%d' % id_cnt
                pic_id = np.random.randint(0, 11)
                pic = cv2.imread('./db_test_pic/vistors/{}.jpg'.format(pic_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(-1)))
            else:
                member_id = np.random.randint(0, 4)
                ID = date + time + '%d' % id_cnt
                name = names[member_id]
                pic = cv2.imread('./db_test_pic/members/{}.jpg'.format(member_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(member_id)))
            id_cnt += 1

    for j in range(8, 10):
        time = "%02d" % j
        cnt = np.random.randint(10, 12)
        for k in range(0, cnt):
            time += "%02d" % np.random.randint(0, 60)
            time += "%02d" % np.random.randint(0, 60)
            v_m = np.random.randint(0, 2)
            if v_m == 0:
                name = 'Vistor'
                ID = date + time + '%d' % id_cnt
                pic_id = np.random.randint(0, 11)
                pic = cv2.imread('./db_test_pic/vistors/{}.jpg'.format(pic_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(-1)))
            else:
                member_id = np.random.randint(0, 4)
                ID = date + time + '%d' % id_cnt
                name = names[member_id]
                pic = cv2.imread('./db_test_pic/members/{}.jpg'.format(member_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(member_id)))
            id_cnt += 1

    for j in range(10, 18):
        time = "%02d" % j
        cnt = np.random.randint(5, 15)
        for k in range(0, cnt):
            time += "%02d" % np.random.randint(0, 60)
            time += "%02d" % np.random.randint(0, 60)
            v_m = np.random.randint(0, 2)
            if v_m == 0:
                name = 'Vistor'
                ID = date + time + '%d' % id_cnt
                pic_id = np.random.randint(0, 11)
                pic = cv2.imread('./db_test_pic/vistors/{}.jpg'.format(pic_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(-1)))
            else:
                member_id = np.random.randint(0, 4)
                ID = date + time + '%d' % id_cnt
                name = names[member_id]
                pic = cv2.imread('./db_test_pic/members/{}.jpg'.format(member_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(member_id)))
            id_cnt += 1

    for j in range(18, 20):
        time = "%02d" % j
        cnt = np.random.randint(10, 20)
        for k in range(0, cnt):
            time += "%02d" % np.random.randint(0, 60)
            time += "%02d" % np.random.randint(0, 60)
            v_m = np.random.randint(0, 2)
            if v_m == 0:
                name = 'Vistor'
                ID = date + time + '%d' % id_cnt
                pic_id = np.random.randint(0, 11)
                pic = cv2.imread('./db_test_pic/vistors/{}.jpg'.format(pic_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(-1)))
            else:
                member_id = np.random.randint(0, 4)
                ID = date + time + '%d' % id_cnt
                name = names[member_id]
                pic = cv2.imread('./db_test_pic/members/{}.jpg'.format(member_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(member_id)))
            id_cnt += 1

    for j in range(20, 23):
        time = "%02d" % j
        cnt = np.random.randint(5, 10)
        for k in range(0, cnt):
            time += "%02d" % np.random.randint(0, 60)
            time += "%02d" % np.random.randint(0, 60)
            v_m = np.random.randint(0, 2)
            if v_m == 0:
                name = 'Vistor'
                ID = date + time + '%d' % id_cnt
                pic_id = np.random.randint(0, 11)
                pic = cv2.imread('./db_test_pic/vistors/{}.jpg'.format(pic_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(-1)))
            else:
                member_id = np.random.randint(0, 4)
                ID = date + time + '%d' % id_cnt
                name = names[member_id]
                pic = cv2.imread('./db_test_pic/members/{}.jpg'.format(member_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(member_id)))
            id_cnt += 1

    for j in range(23, 24):
        time = "%02d" % j
        cnt = np.random.randint(0, 3)
        for k in range(0, cnt):
            time += "%02d" % np.random.randint(0, 60)
            time += "%02d" % np.random.randint(0, 60)
            v_m = np.random.randint(0, 2)
            if v_m == 0:
                name = 'Vistor'
                ID = date + time + '%d' % id_cnt
                pic_id = np.random.randint(0, 11)
                pic = cv2.imread('./db_test_pic/vistors/{}.jpg'.format(pic_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(-1)))
            else:
                member_id = np.random.randint(0, 4)
                ID = date + time + '%d' % id_cnt
                name = names[member_id]
                pic = cv2.imread('./db_test_pic/members/{}.jpg'.format(member_id))
                sql = "INSERT INTO entry_record(ID, EntryTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                sql = "INSERT INTO departure_record(ID, DepartureTime, Picture, Name, Date, Member) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (ID, time, pymysql.Binary(pic), name, date, str(member_id)))
            id_cnt += 1
    conn.commit()
    print(date + "finished!")
    








