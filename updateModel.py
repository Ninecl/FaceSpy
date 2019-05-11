import os
import knn_classifier
import datetime
import time


def main():
    while True:
        now = datetime.datetime.now()
        if now.hour == 1:
            knn_classifier.train_knn_model("./imgs/train_imgs", "./models/20180402-114759.pb", "./models/knn_classifier.pkl")
        time.sleep(3600)


if __name__ == "__main__":
    main()

