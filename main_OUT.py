# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import Process, Queue, Lock, Event

import video_recieve
import video_detect
import video_process
import cnt
import os


# 创建读取摄像头视频流的队列
rightFrameQue = Queue()
middleFrameQue = Queue()
leftFrameQue = Queue()
# 创建检测人脸进程的队列
rightVideoQue = Queue()
middleVideoQue = Queue()
leftVideoQue = Queue()
# 创建识别人脸进程的队列
alreadyQue = Queue()
# 定义一个进程锁，alreadyInList每次只能有一个进程访问
mutex = Lock()
# 创建两个事件，用来调度人脸识别函数
middleVideoEvent = Event()
leftVideoEvent = Event()
rightVideoEvent = Event()
# 创建两个事件，用来调度人脸检测函数
frameEvent = Event()



def main():

    # 首先创建进程
    pRecieveMiddle = Process(target=video_recieve.recieve, args=(middleFrameQue, "192.168.1.105"))
    pDetectMiddle = Process(target=video_detect.detect, args=(middleFrameQue, middleVideoQue, "OUT", middleVideoEvent))
    pProcessMiddle = Process(target=video_process.process, args=(middleVideoQue, alreadyQue, "OUT", "MIDDLE", middleVideoEvent))
    pRecieveLeft = Process(target=video_recieve.recieve, args=(leftFrameQue, "192.168.1.101"))
    pDetectLeft = Process(target=video_detect.detect, args=(leftFrameQue, leftVideoQue, "OUT", leftVideoEvent))
    pProcessLeft = Process(target=video_process.process, args=(leftVideoQue, alreadyQue, "OUT", "LEFT", leftVideoEvent))
    pRecieveRight = Process(target=video_recieve.recieve, args=(rightFrameQue, "192.168.1.104"))
    pDetectRight = Process(target=video_detect.detect, args=(rightFrameQue, rightVideoQue, "OUT", rightVideoEvent))
    pProcessRight = Process(target=video_process.process, args=(rightVideoQue, alreadyQue, "OUT", "RIGHT", rightVideoEvent))
    pCnt = Process(target=cnt.collect_cnt_person, args=(alreadyQue, "OUT"))
    # 启动进程
    pRecieveMiddle.start()
    pDetectMiddle.start()
    pProcessMiddle.start()
    pRecieveLeft.start()
    pDetectLeft.start()
    pProcessLeft.start()
    pRecieveRight.start()
    pDetectRight.start()
    pProcessRight.start()
    pCnt.start()
    # join进程
    pRecieveMiddle.join()
    pDetectMiddle.join()
    pProcessMiddle.join()
    pRecieveLeft.join()
    pDetectLeft.join()
    pProcessLeft.join()
    pRecieveRight.join()
    pDetectRight.join()
    pProcessRight.join()
    pCnt.join()


if __name__ == '__main__':
    main()

