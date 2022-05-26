本科毕业设计项目附件说明
打包日期：2022年5月15日

研究内容，基于yolov5s算法，在pytorch环境下，设计了鱼脑实时目标追踪算法的设计，最终推理速度为3.23ms，精度误差为±1像素。

文件功能介绍：
1. predict文件夹为本项目测试集的预测结果，共7409张图片。

2. test文件夹为本项目的测试集，共7409张图片，最后附有labels.txt文件，为测试及图片的标注文件。（训练集和验证集太大，没有包含在附件内）

3. yolov5文件夹为当前Github下最新版的YOLOv5开源项目包，里面有项目需要用到的工具文件。

4. Zebrafish Object Detection TensorRT Project 为鱼脑检测算法的TensoRT优化项目，其基于C++语言进行实现，详细信息参考文件夹内的reame.txt。

5. best_OD_weight.onnx 为本项目数据集训练之后的onnx权重文件。

6. best_OD_weight.pth 为本项目数据集训练之后的pth权重文件。

7. my_dataset.py 为自定义的读取当前数据格式而重写的dataloader文件，用于本项目的数据读取。

8. my_OD_model.py 为鱼脑检测算法模型的设计文件。

9. my_OD_train.py 为鱼脑检测算法的训练文件。

10. my_OD_predict.py 为鱼脑检测算法的测试集预测文件。

11. pth_to_onx.py 为pth权重文件转onnx权重文件实现。

12. split_set.py 为本项目的数据划分文件，用于将整个数据集划分为训练集，验证集和测试集。

注：上述所有文件均可在作者自己的设备上成功运行。
设备环境为：
NVIDIA驱动版本为：511.79
cuda版本为：11.3
pytorch版本为：1.11.0
opencv版本为：4.55
python版本：3.8
windows10环境
GPU：MX250
