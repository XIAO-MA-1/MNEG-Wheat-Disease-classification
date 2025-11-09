# MNEG-Wheat-Disease-classification
官方 PyTorch 实现 | 论文《MNEG: Research on the Deep Learning Model with Fused Hierarchical Dual Attention Mechanism in Wheat Leaf Disease Detection》（投稿阶段）  
# 1.项目简介
小麦是全球核心粮食作物，但叶片病害（白粉病、斑枯病、叶枯病、叶锈病等）导致全球年产量损失 10%-25%，传统人工识别效率低、实验室检测周期长，现有深度学习模型难以平衡轻量化部署与高精度识别需求。  
本项目提出轻量级模型 MNEG，基于改进 MobileNetV2 架构，融合 EMA 与 GAM 双注意力机制，构建 “低层次特征筛选 - 中高层次特征增强” 的阶梯式优化路径，在保证模型轻量化的同时，实现小麦叶片病害的精准识别。核心成果包括：  
- 自建数据集上识别准确率达**98.55%**，显著优于 MobileNetV3、GhostNet 等主流模型；
- 跨作物泛化能力强，在 PlantVillage 番茄病害数据集上（无调参）准确率达**89.89%**；
- 模型参数量仅 2.62M，单张图片推理速度 16.4ms，适配边缘设备部署。
# 2.核心创新
- 轻量化架构优化：移除 MobileNetV2 中冗余的 64 通道倒残差组，参数量减少 25.1%，避免过拟合的同时降低边缘设备部署门槛；  
- 阶梯式双注意力融合：初始卷积层后引入 EMA 注意力（筛选低层次关键特征、抑制噪声），倒残差块中嵌入 GAM 注意力（强化中高层语义特征），二者协同互补；  
- 强泛化能力设计：不依赖小麦特异性特征，通过 “去冗余 + 通用注意力” 架构，可迁移至番茄、玉米等其他作物病害识别场景。
# 3.数据集说明
## 3.1 自建小麦叶片病害数据集 （数据集已随项目上传至仓库中（Dataset-classification），无需额外下载）
- 样本规模：7230张图像，涵盖5类小麦图片（白粉病、斑枯病、叶枯病、叶锈病、健康叶片）；
- 数据来源：河南省小麦种植基地实地采集+GAN生成补充，含不同光照、角度、背景变化；
- 划分比例：训练集60%（4338张）、验证集20%（1446张）、测试集20%（1446张），且数据集的划分由代码统一划分；
## 3.2 开源数据集（泛化验证）
- 采用PlantVillage数据集的番茄病害子集，共11570张图像，含5类病害+健康样本；
- 划分比例: 训练集60%（6942张）、验证集20%（2314张）、测试集20%（2314张）；
- 下载链接：https://aistudio.baidu.com/datasetdetail/57525。
## 3.3 数据集结构
仓库中的Dataset-classification文件夹中含有我们用于模型训练/测试的图像：  
```
Dataset-classification/
   ├──data_wheat/
      ├──text/（验证集，格式与训练集相同）
      ├──val/（测试集，格式与训练集相同）
      ├──train/（训练集）
         ├── 枯萎病 / # 小麦枯萎病叶片图像（病斑特征：叶片枯萎、变色，严重时大面积干枯，影响植株生理功能）  
         ├── 白粉病 / # 小麦白粉病叶片图像（病斑特征：叶片表面覆盖白色粉状霉层，阻碍气体交换与光合作用）  
         ├── 斑枯病 / # 小麦斑枯病叶片图像（病斑特征：初期为褐色小斑点，随病情发展扩大并连接成片）  
         ├── 叶锈病 / # 小麦叶锈病叶片图像（病斑特征：产生锈色孢子堆，初期黄色或橙色，后期深褐色）  
         └── 健康植株 / # 健康小麦植株图像（特征：叶片鲜绿、形态舒展有光泽，无病害或虫害迹象  
```

# 4.实验环境配置
## 4.1依赖安装
推荐使用Anaconde创建虚拟虚拟环境安装，确保依赖版本匹配（避免兼容性问题，尤其适配Pytorch=2.1.1+cu118）
```
# 1. 创建并激活虚拟环境
conda creat -n D:\environments\three python=3.9
conda activate D:\environemnts\three

# 2. 安装PyTorch、TorchVision和Torchaudio（需适配CUDA版本）
pip install D:\whl\torch-1.12.0+cu113-cp39-cp39-win_amd64.whl
pip install D:\whl\torchvision-0.13.0+cu113-cp39-cp39-win_amd64.whl
pip install D:\whl\torchaudio-0.12.0+cu113-cp39-cp39-win_amd64.whl

# 3. 安装其他依赖库（数据处理、可视化、模型工具等）
pip install numpy~=1.26.4 matplotlib~=3.5.1 opencv-python~=4.8.0.76
pip install pandas~=1.2.4 pillow~=11.2.1
pip install tqdm~=4.67.1 
```
## 4.2硬件要求
| 配置项                   | 详情                                      |
|-------------------------|-------------------------------------------|
| Operating system        | Windows 11(64-bit)                        |
| Processor               | 12th Gen Intel(R) Core(TM) i9-12900H      |
| Graphics card           | NVIDIA GeForce RTX 3060 Laptop GPU        |
| Memory                  | 16GB                                      |
| Deep learning framework | Pytorch                                   |
| Editor                  | Pycharm 2023.3                            |
| Programming language    | Python3.9                                 |
# 5.实验结果
MNEG模型与主流的深度学习模型在小麦病害分类任务上的性能对比如下；
| 模型       | 准确率  | 精确率  | 召回率  | F1-score | 特异性  |
|------------|---------|---------|---------|----------|---------|
| MNEG       | 98.55%  | 99.24%  | 98.43%  | 98.83%   | 99.24%  |
| MobileNetV3| 71.99%  | 71.10%  | 70.57%  | 70.83%   | 93.09%  |
| GhostNet   | 92.53%  | 92.79%  | 91.87%  | 92.33%   | 98.16%  |
| MobileVIT  | 77.18%  | 75.61%  | 74.91%  | 75.26%   | 94.35%  |
# 6.代码使用说明
## 6.1模型训练
运行```train.py```脚本启动训练，支持通过参数调整训练配置，示例如下：
```
python train.py \
   --data_dir ./Dataset-classification/
   --epoch 100\
   --batch_size 64\
   --lr  1e-3\
   --weight_decay 1e-5 \
   --save_dir ./weights \
   --device cuda:0 \
```
## 关键参数说明:

