# -*- coding:utf-8 -*-
"""
作者：tgd
日期：2022年03月17日
"""
import os
import torch
from torchvision import transforms
from my_dataset import my_dataset
from my_OD_model import ODModel
import cv2 as cv
from tqdm import tqdm
import sys
import time

# select computer device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
# initialize the model weight
OD_model = ODModel(pretrained=False)
weights_path = "./best_OD_weight.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

OD_model.load_state_dict(torch.load(weights_path))

# run on device
OD_model.to(device)


# data transforms
data_transform = transforms.Compose(
        [ transforms.Resize((320, 320)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# set cpu work num and data batch size
nw = 0
batch_size = 1

# set image path
save_path = "./predict"
test_path = "./test"

# load data and split it
test_data = my_dataset(path=test_path, data_transforms=data_transform, tar_size=320)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=nw)

for step, (image, label) in enumerate(test_data_loader):
    s = time.time()
    OD_model(image.to(device))
    e = time.time()
    print(e - s)





# predict test images
predict_bar = tqdm(test_data_loader, file=sys.stdout)
test_error = 0
for step, (test_image, test_label) in enumerate(predict_bar):
    OD_model.eval()
    with torch.no_grad():
        # predict the coordinate of zebra fish in the test image
        output = OD_model(test_image.to(device))
        predict_bar.desc = "valid epoch[{}/{}]".format(0, 1)
        test_error += sum(abs(output - test_label.to(device)))

    # display predict coordinate data
    # print_res = "real label:{}\n predict label:{}".format(test_label, output)
    # print(print_res)

    # trans_transform the output coordinate
    output = output*360/320

    # read the test image original path
    test_path = test_data.img_name[step]

    # display predict coordinate on the original test image
    img = cv.imread(test_path)
    out_point = list(output[0].cpu().numpy())
    cv.circle(img, [round(out_point[0]), round(out_point[1])], 2, color=(255, 0, 0), thickness=1)
    cv.circle(img, [round(out_point[2]), round(out_point[3])], 2, color=(0, 0, 255), thickness=1)
    cv.imshow('window_title', img)

    # wait to display the zebra fish, press any to continue
    cv.waitKey()

    # set the save path of the predict image
    image_path = os.path.join(save_path, test_path.split('\\')[1])

    # save the image
    # cv.imwrite(image_path, img)

test_error_average = test_error/len(test_data)
print(test_error_average)