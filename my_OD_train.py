# -*- coding:utf-8 -*-
"""
作者：tgd
日期：2022年03月17日
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from my_dataset import my_dataset
from my_OD_model import ODModel

# select computer device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# set the path of using train/val data path
data_path = "E:/data"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")

# set the data transform parameters
batch_size = 20
nw = 0

data_transform = {
    "train": transforms.Compose([transforms.Resize(320),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 transforms.ConvertImageDtype(torch.float32)]),
    "val": transforms.Compose([transforms.Resize(320),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.ConvertImageDtype(torch.float32)])}

# extract the train data
train_data = my_dataset(path=train_path, data_transforms=data_transform['train'], tar_size=320)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nw)

# extract the val data
val_data = my_dataset(path=val_path, data_transforms=data_transform["val"], tar_size=320)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=50, num_workers=nw)


# create model
OD_model = ODModel(pretrained=False)
weights_path = "./best_OD_weight.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
OD_model.load_state_dict(torch.load(weights_path))

# run on device
OD_model.to(device)

# define the loss and loss algorithm
Loss_fun = nn.SmoothL1Loss()

# define the optimal algorithm
params = [p for p in OD_model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.000001)

# set the epoch num, the best weight path and initialize the best_error
epoch_num = 100
best_weight_path = "./new_best_OD_weight.pth"
best_error = 1000

# start train
for epoch in range(epoch_num):

    # train start
    OD_model.train()
    train_error = 0
    train_bar = tqdm(train_data_loader, file=sys.stdout)
    for step, (train_images, train_labels) in enumerate(train_bar):
        OD_model.zero_grad()
        train_step_output = OD_model(train_images.to(device))
        train_step_loss = Loss_fun(train_step_output, train_labels.to(device))
        train_error += sum(abs(train_step_output - train_labels.to(device)))
        train_step_loss.backward()
        optimizer.step()
        train_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epoch_num)

    # val start
    OD_model.eval()
    val_error = 0
    with torch.no_grad():
        val_bar = tqdm(val_data_loader)
        for step, (val_images, val_labels) in enumerate(val_bar):
            OD_model.zero_grad()
            val_step_output = OD_model(val_images.to(device))
            val_step_loss = Loss_fun(val_step_output, val_labels.to(device))
            val_error += sum(abs(val_step_output - val_labels.to(device)))
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epoch_num)
            # write the weight of the best error
            if val_step_loss < best_error:
                best_error = val_step_loss
                torch.save(OD_model.state_dict(), best_weight_path)


    # output the train information
    train_mean_error = train_error / len(train_data)
    val_mean_error = val_error / len(val_data)

    print('[epoch {}] was finished:'.format(epoch + 1))
    print('train mean error is:'.format())
    print(train_mean_error)
    print('val mean error is:'.format())
    print(val_mean_error)
    print('\n\n')


