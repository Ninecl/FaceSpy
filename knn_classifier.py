from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
import scipy
import PIL
from sklearn.svm import SVC

def get_paths_and_labels_and_classes():
    paths = []
    labels = []
    standard_imgs_path = "./imgs/train_imgs/standard_imgs/"
    update_imgs_path = "./imgs/train_imgs/update_imgs/"
    classes = len(os.listdir(standard_imgs_path))

    # 将每个人的标准照片加入paths和labels中
    for i in range(classes):
        standard_img_path = standard_imgs_path + "{}/".format(i)
        for j in range(len(os.listdir(standard_img_path))):
            paths.append(standard_img_path + "{}.jpg".format(j))
            labels.append(i)

    # 将每个人的更新照片(如果有的话)加入paths和labels中
    for i in range(classes):
        update_img_path = update_imgs_path + "{}/".format(i)
        if os.path.exists(update_img_path):
            for j in range(len(os.listdir(update_img_path))):
                paths.append(update_img_path + "{}.jpg".format(j))
                labels.append(i)
        
    return paths, labels, classes


def train_knn_model(data_dir, model, classifier_filename,
        image_size = 160, batch_size = 90,
        seed = 666, use_split_dataset = False):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            paths, labels, classes = get_paths_and_labels_and_classes()
            print(paths)
            print(labels)
            
            print('Number of classes: %d' % classes)
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(classifier_filename)

            model = {"embs": emb_array, "labels": labels, "classes": classes}
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump(model, outfile)
            with open(classifier_filename_exp, 'rb') as infile:
                model = pickle.load(infile)
            print(model)
            


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in datzhixingaset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)


if __name__ == "__main__":
    train_knn_model("./imgs/train_imgs", "./models/20180402-114759.pb", "./models/try.pkl") 
