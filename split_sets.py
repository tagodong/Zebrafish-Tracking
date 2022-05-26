# -*- coding:utf-8 -*-
"""
作者：tgd
日期：2022年03月16日
"""
import os
from shutil import copy, rmtree
import random

# split data to train sets, val sets and test sets.
train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

data_path = "E:/data"
data_train_path = os.path.join(data_path, "train")
data_val_path = os.path.join(data_path, "val")
data_test_path = os.path.join(data_path, "test")

# create dirs
if not os.path.exists(data_train_path):
    os.makedirs(data_train_path)
if not os.path.exists(data_val_path):
    os.makedirs(data_val_path)
if not os.path.exists(data_test_path):
    os.makedirs(data_test_path)

# split data start
images_sets = [images_set for images_set in os.listdir(data_path) if (images_set not in ['train', 'val', 'test'])]
last_image_num = 0
current_num = 0
proceed_num = 0
for images_set in images_sets:
    proceed_num += 1
    images_path = os.path.join(data_path, images_set)
    with open(os.path.join(images_path, 'labels.txt'), 'r') as f_read, \
            open(os.path.join(data_train_path, 'labels.txt'), 'a+') as f_train_write,\
            open(os.path.join(data_val_path, 'labels.txt'), 'a+') as f_val_write,\
            open(os.path.join(data_test_path, 'labels.txt'), 'a+') as f_test_write:
        images = os.listdir(images_path)
        for image in images:
            if image.split('.')[1] == 'png':
                label_text = f_read.readline().split(',')
                image_name = str(int(image.split('.')[0]) + last_image_num) + '.png'
                label_text[0] = str(int(image.split('.')[0]) + last_image_num)
                image_path = os.path.join(images_path, image)

                # using the random num to decide the direction
                rand_index = random.random()
                if rand_index <= train_prop:
                    new_path = os.path.join(data_train_path, image_name)
                    f_train_write.write(','.join(label_text))
                elif rand_index <= (train_prop + val_prop):
                    new_path = os.path.join(data_val_path, image_name)
                    f_val_write.write(','.join(label_text))
                else:
                    new_path = os.path.join(data_test_path, image_name)
                    f_test_write.write(','.join(label_text))

                # save the data
                copy(image_path, new_path)
        last_image_num += len(images)

    # display current proceed
    print("finished proceed is [{}/{}]".format(proceed_num, len(images_sets), end = ""))

# display the final information
print("train data ratio: {}".format(len(os.listdir(data_train_path))))
print("val data ratio: {}".format(len(os.listdir(data_val_path))))
print("test data ratio: {}".format(len(os.listdir(data_test_path))))
