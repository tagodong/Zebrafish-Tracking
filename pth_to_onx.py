# -*- coding:utf-8 -*-
"""
作者：tgd
日期：2022年03月22日
"""

import onnx
import numpy as np
import torch.onnx
from my_OD_model import ODModel


weight_file = './best_OD_weight.pth'

onnx_file_name = './best_OD_weight.onnx'


model = ODModel(pretrained=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加快模型训练的效率
model_dict = model.state_dict()
pretrained_dict = torch.load(weight_file, map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()

IN_IMAGE_H = 320
IN_IMAGE_W = 320

dummy_input1 = torch.randn(1, 3, IN_IMAGE_H, IN_IMAGE_W, requires_grad=True)
input_names = ["x"]  # onnx输入接口的名字，需要与模型输入结果对应
output_names = ["output"]  # onnx输出接口的名字，需要与模型输出结果对应
dynamic_axes = {"x": {0: "batch_size"}, "output": {0: "batch_size"}}

torch.onnx.export(model,
                  dummy_input1,
                  onnx_file_name,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=input_names,
                  output_names=output_names,
                  #dynamic_axes=dynamic_axes,
                  )

net = onnx.load(onnx_file_name)         #加载onnx 计算图
onnx.checker.check_model(net)           # 检查文件模型是否正确
onnx.helper.printable_graph(net.graph)  #输出onnx的计算图