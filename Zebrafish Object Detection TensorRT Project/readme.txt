本科毕业设计项目附件：Zebrafish Object Detection TensorRT Project
打包日期：2022年5月15日

Zebrafish Object Detection TensorRT Project 是鱼脑检测算法的TensorRT优化的C++工程项目

使用说明：
直接打开TensorRT.sln，运行mian.cpp和Zebrafish_OD_TRT.cpp的代码，该部分代码为项目实现的主体。

其他文件为TensorRT的工具包，可以自行探索。

本项目将依赖的环境全部进行了打包，放在了lean文件夹下，本系统的运行环境不会干扰电脑固有的运行环境。

该项目经本人检验，确认可以在以下环境中运行：
1. Visual Studio 版本为：2017
2. windows10 系统
3. Cuda 版本为11.3
4. GPU的算力为6.1（算力需要自己在NVIDIA官网查询对应GPU型号的算力）
查询官网为：https://developer.nvidia.com/zh-cn/cuda-gpus#compute

注：
若运行环境要更换Cuda的版本，需要进行如下操作：
用记事本打开 TensorRT.vcxproj，将其中的CUDA 11.3.props和CUDA 11.3.targets更换为运行环境的版本。

注：
若GPU算力不为6.1，需要进行如下操作：
用记事本打开 TensorRT.vcxproj，将其中的compute_61,sm_61更换为运行环境的GPU算力。