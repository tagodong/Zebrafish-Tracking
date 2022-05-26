# -*- coding:utf-8 -*-
"""
作者：tgd
日期：2022年03月17日
"""
import torch
from yolov5.models.yolo import Model
import torch.nn as nn
from yolov5.models.common import Conv, C3, SPPF, Concat
from yolov5.utils import torch_utils
from yolov5.utils.general import intersect_dicts


class ODModel(nn.Module):
    def __init__(self, class_num=4, pretrained=True):
        super(ODModel, self).__init__()
        self.build_model(class_num)
        torch_utils.initialize_weights(self)
        if pretrained:
            self._initialize_weights()

    def build_model(self, class_num):
        # output channels
        self.output_ch = class_num
        # backbone
        self.Conv_1 = Conv(c1=3, c2=32, k=6, s=2, p=2)
        self.Conv_2 = Conv(c1=32, c2=64, k=3, s=2)
        self.C3_1 = C3(c1=64, c2=64)
        self.Conv_3 = Conv(c1=64, c2=128, k=3, s=2)
        self.C3_2 = C3(c1=128, c2=128, n=2)
        self.Conv_4 = Conv(c1=128, c2=256, k=3, s=2)

        self.C3_3 = C3(c1=256, c2=256, n=3)
        self.Conv_5 = Conv(c1=256, c2=512, k=3, s=2)
        self.C3_4 = C3(c1=512, c2=512)
        self.SPPF = SPPF(c1=512, c2=512, k=5)

        # head
        self.Conv_6 = Conv(c1=512, c2=256, k=1, s=1)

        #self.down = nn.MaxPool2d(kernel_size=3, stride=2)
        # have a Concat

        self.C3_5 = C3(c1=256, c2=256)
        self.Conv_7 = Conv(c1=256, c2=128, k=3, s=2) # 256,5,5

        # Detect
        self.Con_8 = nn.Conv2d(in_channels=128, out_channels=self.output_ch, kernel_size=5)
        self.l_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # backbone
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.C3_1(x)
        x = self.Conv_3(x)
        x = self.C3_2(x)
        #y = self.Conv_4(x)
        x = self.Conv_4(x)
        x = self.C3_3(x)
        x = self.Conv_5(x)
        x = self.C3_4(x)
        x = self.SPPF(x)

        # head
        x = self.Conv_6(x)
        #y_down = self.down(y)
        #y_cat = Concat([x, y_down])
        x = self.C3_5(x)
        x = self.Conv_7(x)

        # detect
        x = self.Con_8(x)
        x = self.l_relu(x)
        return x.view(-1, 4)

    # initialize weights of the model with yolov5s' backbone
    def _initialize_weights(self):
        weights = './yolov5/weights/yolov5s.pt'
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint
        dic = {}
        dic['backbone'] = ckpt['model'].yaml['backbone']
        self.load_state_dict(dic, strict=False)
