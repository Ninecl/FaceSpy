import pymysql
import cv2
import os


"""
conn = pymysql.connect("39.98.90.118", "zyf", "zyf123456", "face_recognition")
cursor = conn.cursor()

sql = "select Picture from members where ID = %s" % str(0)
cursor.execute(sql)
data = cursor.fetchone()[0]

imgout = open("./1.jpg", 'wb')
imgout.write(data)
conn.close()
"""


def main():
    path = "../members/dataset/"
    dirs = os.listdir(path)
    for p in dirs:
        imgs = os.listdir(path + p)
        print(len(imgs))


if __name__ == "__main__":
    main()

