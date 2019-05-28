# -*- coding: utf-8 -*-
import json
import requests

head = {'Content-Type': 'application/json'}

def staff_pull(url, deviceSn, ver, isInit, callback):
    """
    url: [string]地址
    deviceSn: [String]设备序列号
    ver: [String]版本号
    isInit: [int]是否初始化
    """
    datas = {
            "deviceSn": deviceSn,
            "ver": ver,
            "isInit": isInit}
    r = requests.post(url, data=json.dumps(datas), headers=head)
    response = json.loads(r.text)
    if response["code"] == 0:
        msg = response["msg"][0]
        ID = msg["id"]
        personSn = msg["personSn"]
        personName = msg["personName"]
        personPhoto = msg["personPhoto"]
        Type = msg["type"]
        callback(url, ID)
        print(ID, personSn, personName, personPhoto, Type)
        return ID, personSn, personName, personPhoto, Type
    else:
        print("pull failed")


def staff_pull_callback(url, ID):
    """
    ID: [String]执行过程id
    该函数作为staff_pull的回调函数
    """
    datas = {"id": ID}
    r = requests.post(url, data=json.dumps(datas), headers=head)


def log_upload(url, image, createTime, personSn):
    """
    url: [String]请求地址
    image: [base64]base64照片
    createTime: [String]创建事件
    personSn: [String]用户id
    """
    datas = {
            "image": image,
            "createTime": createTime,
            "personSn": personSn}
    r = requests.post(url, data=json.dumps(datas), headers=head)
    print(r.text)


if __name__ == "__main__":
    staff_pull("http://127.0.0.1:8000", "1", "2", "3", staff_pull_callback)
