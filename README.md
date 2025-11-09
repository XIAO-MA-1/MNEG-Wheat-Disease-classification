# MNEG-Wheat-Disease-classification
官方 PyTorch 实现 | 论文《MNEG: Research on the Deep Learning Model with Fused Hierarchical Dual Attention Mechanism in Wheat Leaf Disease Detection》（投稿阶段）
# 1.项目简介
小麦是全球核心粮食作物，但叶片病害（白粉病、斑枯病、叶枯病、叶锈病等）导致全球年产量损失 10%-25%，传统人工识别效率低、实验室检测周期长，现有深度学习模型难以平衡轻量化部署与高精度识别需求。  
本项目提出轻量级模型 MNEG，基于改进 MobileNetV2 架构，融合 EMA 与 GAM 双注意力机制，构建 “低层次特征筛选 - 中高层次特征增强” 的阶梯式优化路径，在保证模型轻量化的同时，实现小麦叶片病害的精准识别。核心成果包括：  
- 自建数据集上识别准确率达**98.55%**，显著优于 MobileNetV3、GhostNet 等主流模型；
- 跨作物泛化能力强，在 PlantVillage 番茄病害数据集上（无调参）准确率达**89.89%**；
- 模型参数量仅 2.62M，单张图片推理速度 16.4ms，适配边缘设备部署。
