# 简介
本仓库提供了一个基于 PyTorch 的 [UNet](https://arxiv.org/pdf/1505.04597v1) 网络实现，对眼底视网膜血管和红细胞数据集进行图像语义分割。
UNet 是一种流行的卷积神经网络架构，特别适用于生物医学图像分割等任务。
![Image](https://github.com/user-attachments/assets/a25ec80d-9802-4b02-9724-d56f2e2ad8fd)
此实现包含了模型定义、数据加载、训练评估和推理的基本框架。

## 目录结构：
### unet/
### ├── README.md             #本文件
### ├── data/                 #数据集文件夹
### │   ├── BCCD Dataset with mask #红细胞数据集文件夹
### │   ├── DRIVE        #眼底视网膜血管数据集文件夹
### ├── Datasets/               #数据预处理和加载脚本
### │   ├── BloodCeilDataset.py   #红细胞数据集数据预处理脚本
### │   ├── vesselDataset.py    #眼底视网膜血管数据集预处理脚本
### ├──Unet.py           #模型定义
### ├── Unet_train.py              #训练脚本
### ├── Unet_predict.py           #推理脚本
### ├── my_utils.py                #实用工具函数(数据增强、可视化、评估指标等)
### ├── requirements.txt      #项目依赖包
### └── seg_loss_fn.py             #损失函数

## 环境配置
### 1. 安装依赖
使用 requirements.txt 文件安装项目所需的 Python 包
```  JavaScript
pip install -r requirements.txt
```
### 2. 数据集准备
数据加载脚本包括红细胞和眼底视网膜血管数据预处理脚本，存放在Datasets中
请将您的数据集放置data目录中，对齐数据集目录结构，并在 Datasets中对应数据加载脚本以适应您的数据格式
![Image](https://github.com/user-attachments/assets/585b2aa1-ab99-49bc-abaf-30d43a5d4a93)
### [DRIVE.tar数据集下载](https://pan.baidu.com/s/1E90vjqRItNTGjjPmXEQAsQ?pwd=ueye) 




![Image](https://github.com/user-attachments/assets/5df4f01c-f7c2-429d-8031-58d1f2402433)
### [BCCD Dataset with mask.tar数据集下载](https://pan.baidu.com/s/1tKrI1qs6TeKn3iV00oSu-w?pwd=ueye) 

# 训练模型

### --max_epoch：训练的轮数。
### --batch_size：每批处理的数据量。
### --lr：学习率。
### --loss_function：损失函数，默认交叉熵损失函数
### 运行 Unet_train.py 脚本开始训练模型(在该文件中需手动修改超参，不能通过命令行)

# 模型推理
### 给定图像文件存放目录，可以逐个图像前景分割抠图，下图是训练过程中相应验证集的评估结果

![image](https://github.com/user-attachments/assets/6b498a11-de93-4099-bbd3-8f238b390644)
![image](https://github.com/user-attachments/assets/fff564d3-6d90-47e7-a69f-8bb96ac8dc50)

### 或对给定视频文件名，可以对视频进行图像前景抠图

## [更多详见B站](https://www.bilibili.com/video/BV1xP6LYoE6x/?vd_source=38cfb9337bd66ed55074fc447ec9837d)
### 希望这个 README 文件能帮助您快速上手和理解本仓库中的 UNet PyTorch 实现。祝您使用愉快！
### 如果您有任何问题或建议，请通过 GitHub Issues 与我们联系

