import os
import sys
import json
import torch

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchvision import transforms, datasets
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 64
    epochs = 100


    best_loss = float('inf')


    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 缩小裁剪范围
            transforms.RandomHorizontalFlip(p=0.5),  # 明确概率
            transforms.RandomRotation(10),  # 减小旋转角度
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 减弱颜色扰动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), r"D:\Users\18238\Desktop\mobilenet v2 - 副本"))
    image_path = os.path.join(data_root, "data_set", "data_wheat")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, pin_memory=True)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, pin_memory=True)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # create model
    net = MobileNetV2(num_classes=5).to(device)

    # 冻结部分层
    for param in net.features[:-5].parameters():  # 解冻最后几个层
        param.requires_grad = False

    net.to(device)

    # 定义损失函数（添加标签平滑）
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4)  # 减小初始学习率，添加权重衰减

    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    # 混合精度训练初始化
    scaler = GradScaler()  # 梯度缩放器

    best_acc = 0.0
    save_path = r'D:\Users\18238\Desktop\weight\最终模型.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # 训练阶段
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # 混合精度训练
            with torch.cuda.amp.autocast():  # 启用自动混合精度
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))

            # 梯度缩放和反向传播
            scaler.scale(loss).backward()  # 缩放loss以防止FP16下溢
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放因子

            running_loss += loss.item()

            # 计算训练准确率
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 计算平均训练损失和准确率
        train_loss = running_loss / train_steps
        train_accurate = train_acc / train_num

        # 验证阶段
        net.eval()
        val_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_loss /= len(validate_loader)
        val_accurate = acc / val_num

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss, train_accurate, val_loss, val_accurate))

        # 学习率调度
        scheduler.step(val_loss)

        # 早停和模型保存
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_accurate
            #patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_accurate
            }, save_path)


    print(f'Finished Training. Best validation accuracy: {best_acc:.3f}')


if __name__ == '__main__':
    main()