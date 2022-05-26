# -*- coding:utf-8 -*-
"""
作者：tgd
日期：2022年03月16日
"""
# the function was defined that

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class my_dataset(Dataset):
    def __init__(self, path="", data_transforms=None, tar_size=None, loader=default_loader):
        filenames = os.listdir(path)
        real_file_names = filenames[:-1]
        real_file_names.sort(key=lambda x: int(x[:-4]))
        with open(os.path.join(path, filenames[-1])) as f_labels:
            labels = f_labels.readlines()
            labels = [label.strip() for label in labels]
            labels = np.array([np.array([float(item) for item in label.split(',')[1:5]]) for label in labels])
            if tar_size is not None:
                labels = labels * tar_size/360
            labels = torch.from_numpy(labels)
            self.img_label = labels
        self.img_name = [os.path.join(path, real_file_name) for real_file_name in real_file_names]
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))

        return img, label
