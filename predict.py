import os
import json
import csv
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms


from model import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载类别映射
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件 {json_path} 不存在"
    with open(json_path, 'r', encoding='utf-8') as f:
        class_indict = json.load(f)
    class_names = list(class_indict.values())

    # 创建反向映射字典
    label_map = {v: k for k, v in class_indict.items()}

    # 创建模型
    model = MobileNetV2(num_classes=len(class_indict))
    model = model.to(device)

    # 加载模型权重
    model_weight_path = r"D:\Users\18238\Desktop\weight\text.pth"
    assert os.path.exists(model_weight_path), f"权重文件 {model_weight_path} 不存在"

    # 修复权重加载问题
    checkpoint = torch.load(model_weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        model_weights = checkpoint['state_dict']
    else:
        model_weights = checkpoint

    # 兼容处理权重加载
    try:
        model.load_state_dict(model_weights)
    except RuntimeError as e:
        # 尝试部分匹配键名
        model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}
        model.load_state_dict(model_weights, strict=False)
        print("警告：部分权重未完全匹配，使用非严格加载模式")

    model.eval()
    print("模型权重加载成功")

    # 处理图片文件夹
    folder_path = r"D:\Users\18238\Desktop\mobilenet v2 - 副本\data_set\predict"
    assert os.path.exists(folder_path), f"图片文件夹 {folder_path} 不存在"

    # 获取所有图片文件
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("未找到任何图片文件")
        return

    # 统计变量
    correct_count = 0
    total_count = len(image_files)
    results = []

    # 新增：按类别统计
    category_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'errors': defaultdict(int)  # 错误计数器，存储具体次数
    })

    print(f"\n开始处理 {total_count} 张图片...")
    for img_path in image_files:
        try:
            # 加载图片
            img = Image.open(img_path).convert('RGB')
            img_name = os.path.basename(img_path)

            # 预处理
            img_tensor = data_transform(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

            # 预测
            with torch.no_grad():
                output = model(img_tensor)
                output = torch.squeeze(output).cpu()
                probabilities = torch.softmax(output, dim=0)
                pred_idx = torch.argmax(probabilities).item()
                pred_prob = probabilities[pred_idx].item()
                pred_label = class_indict[str(pred_idx)]

            # 解析真实标签
            real_label = os.path.splitext(img_name)[0].split('_')[0]
            real_idx = label_map.get(real_label, None)

            # 统计结果
            is_correct = real_idx is not None and int(real_idx) == pred_idx
            if is_correct:
                correct_count += 1

            # 记录结果
            result = {
                '图片名称': img_name,
                '真实标签': real_label,
                '预测标签': pred_label,
                '置信度': pred_prob,
                '是否正确': '✓' if is_correct else '✗'
            }
            results.append(result)

            # 更新类别统计
            if real_label:
                category_stats[real_label]['total'] += 1
                if is_correct:
                    category_stats[real_label]['correct'] += 1
                else:
                    # 记录错误识别的具体次数
                    category_stats[real_label]['errors'][pred_label] += 1

            # 打印单张结果
            print(
                f"图片: {img_name:<25} 真实: {real_label:<8} 预测: {pred_label:<8} 概率: {pred_prob:.3f} {result['是否正确']}")

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
            results.append({
                '图片名称': os.path.basename(img_path),
                '真实标签': '未知',
                '预测标签': '错误',
                '置信度': 0,
                '是否正确': '❌'
            })

    # 计算并打印总体准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "=" * 80)
    print(f"总体准确率: {accuracy:.2%} ({correct_count}/{total_count})")
    print("=" * 80)

    # 新增：打印类别详细统计（直接显示数字）
    print("\n类别详细统计（总样本数 正确数 错误分布次数）:")
    print("=" * 80)
    print(f"{'类别':<15}{'总样本数':<10}{'正确数':<10}{'错误分布（次数）':<40}")
    print("-" * 80)

    # 按总样本数降序排序
    sorted_categories = sorted(category_stats.items(),
                               key=lambda x: x[1]['total'],
                               reverse=True)

    for category, stats in sorted_categories:
        total = stats['total']
        correct = stats['correct']
        accuracy_val = correct / total if total > 0 else 0

        # 构建错误分布字符串（只显示次数）
        error_str = ""
        # 按错误次数降序排序
        sorted_errors = sorted(stats['errors'].items(),
                               key=lambda x: x[1],
                               reverse=True)
        for error_label, count in sorted_errors:
            error_str += f"{error_label}:{count}次  "

        print(f"{category:<15}{total:<10}{correct:<10}{error_str or '无'}")

    # 新增：最常被错误识别的种类（直接显示次数）
    print("\n错误识别分析:")
    print("=" * 80)

    # 统计所有错误识别
    error_types = defaultdict(int)
    for category, stats in category_stats.items():
        for pred_label, count in stats['errors'].items():
            error_types[pred_label] += count

    # 按错误次数排序
    top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

    if top_errors:
        print("最常被错误识别的种类（次数）:")
        for label, count in top_errors:
            print(f"- {label}: {count} 次")
    else:
        print("- 无错误识别记录")

    # 保存结果到CSV文件
    csv_path = 'prediction_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['图片名称', '真实标签', '预测标签', '置信度', '是否正确']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已保存到: {csv_path}")

    # 新增：保存类别统计到CSV（只包含数字）
    stats_path = 'category_statistics.csv'
    with open(stats_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['类别', '总样本数', '正确数', '错误分布次数']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for category, stats in sorted_categories:
            # 构建错误分布次数字符串
            error_str = ""
            sorted_errors = sorted(stats['errors'].items(),
                                   key=lambda x: x[1],
                                   reverse=True)
            for error_label, count in sorted_errors:
                error_str += f"{error_label}:{count}次 "

            writer.writerow({
                '类别': category,
                '总样本数': stats['total'],
                '正确数': stats['correct'],
                '错误分布次数': error_str or '无'
            })
    print(f"类别统计已保存到: {stats_path}")


if __name__ == '__main__':
    main()