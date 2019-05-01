# -*- coding: utf-8 -*-
import os
import pymysql
import numpy as np
from scipy import misc



def main():
    try:
        conn = pymysql.connect("localhost", "root", "admin123", "face_recognition")
        cursor = conn.cursor()
    except:
        print("Fail to connect DB!")

    f = open("members/members.txt")
    lines = f.readlines()
    f.close()
    sql = "delete from members"
    cursor.execute(sql)
    for line in lines:
        if len(line) == 0:
            continue
        line = line.split()
        ID = int(line[0])
        Name = line[1]
        date = line[2]
        Authority = line[3]
        fp = open('members/members_pics/{}.jpg'.format(ID), 'rb')
        img = fp.read()
        sql = "INSERT INTO members(ID, Name, Picture, RegisterTime, Authority, Inside) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (ID, Name, pymysql.Binary(img), date, Authority, 0))
    conn.commit()


main()
