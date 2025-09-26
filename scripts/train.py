# scripts/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import numpy as np

from data import *
from models.fusion_model import RadarCameraFusionModel
from utils import *


def prepare_targets(batch_data, data_cfg):
    """目标准备函数"""
    batch_size = batch_data['voxels'].shape[0]
    pc_cfg = data_cfg['pointcloud']
    x_range = pc_cfg['x_range']
    y_range = pc_cfg['y_range']

    # BEV网格尺寸
    grid_h, grid_w = 32, 32
    cell_size_x = (x_range[1] - x_range[0]) / grid_w
    cell_size_y = (y_range[1] - y_range[0]) / grid_h

    # 初始化标签
    # 分类标签: [B, H, W] -> 需要调整为 [B, H, W]
    cls_labels = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.long)
    # 回归目标: [B, 7, H, W] -> 需要调整为 [B, H, W, 7]
    reg_targets = torch.zeros(batch_size, grid_h, grid_w, 7)  # x,y,z,w,l,h,theta
    # 方向标签: [B, H, W] -> 需要调整为 [B, H, W]
    dir_labels = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.long)

    # 类别映射
    class_map = {'Car': 0, 'Cyclist': 1, 'Truck': 2}

    for i in range(batch_size):
        if 'labels' in batch_data and batch_data['labels'][i]:
            for label in batch_data['labels'][i]:
                # 提取3D边界框信息
                center_x, center_y, center_z = label['location']

                # KITTI格式的尺寸顺序是(height, width, length)
                height, width, length = label['dimensions']
                rotation = label['rotation_y']

                # 修正网格坐标计算
                # x坐标对应网格的列（width方向），y坐标对应网格的行（height方向）
                grid_x = int((center_x - x_range[0] + cell_size_x/2) / cell_size_x)
                grid_y = int((center_y - y_range[0] + cell_size_y/2) / cell_size_y)
                # 限制在有效范围内
                grid_x = max(0, min(grid_w - 1, grid_x))
                grid_y = max(0, min(grid_h - 1, grid_y))

                # 跳过无效类别
                if label['type'] not in class_map:
                    continue

                class_id = class_map[label['type']]

                # 设置分类标签
                cls_labels[i, grid_y, grid_x] = class_id + 1  # 0保留给背景

                # 计算回归目标：相对偏移 + 绝对尺寸
                # 计算相对于网格中心的偏移量（归一化）
                center_x_in_voxel = (center_x - x_range[0]) % cell_size_x
                center_y_in_voxel = (center_y - y_range[0]) % cell_size_y

                # x方向的偏移（相对于网格中心）
                reg_targets[i, grid_y, grid_x, 0] = (center_x_in_voxel - cell_size_x/2) / cell_size_x


                # y方向的偏移（相对于网格中心）
                reg_targets[i, grid_y, grid_x, 1] = (center_y_in_voxel - cell_size_y/2) / cell_size_y

                # z坐标（绝对高度）
                reg_targets[i, grid_y, grid_x, 2] = center_z

                # 尺寸（height, width, length） - 使用对数尺度便于回归
                reg_targets[i, grid_y, grid_x, 3] = torch.log(torch.tensor(height + 1e-8))  # height
                reg_targets[i, grid_y, grid_x, 4] = torch.log(torch.tensor(width + 1e-8))  # width
                reg_targets[i, grid_y, grid_x, 5] = torch.log(torch.tensor(length + 1e-8))  # length
                # 旋转角度（归一化到[-pi, pi]）
                rotation = rotation % (2 * torch.pi)
                if rotation > torch.pi:
                    rotation -= 2 * torch.pi
                reg_targets[i, grid_y, grid_x, 6] = rotation

                # 方向标签：0表示正向，1表示反向
                # 根据旋转角度判断物体朝向
                dir_labels[i, grid_y, grid_x] = 0 if abs(rotation) < torch.pi / 2 else 1

    # 调整张量形状以匹配模型输出
    # 分类预测需要是 [B, C, H, W]，标签是 [B, H, W]
    reg_targets = reg_targets.permute(0, 3, 1, 2)  # [B, H, W, 7] -> [B, 7, H, W]

    # 方向预测需要是 [B, 2, H, W]，但标签是 [B, H, W]

    return {
        'cls_labels': cls_labels,  # [B, H, W]
        'reg_targets': reg_targets,  # [B, 7, H, W]
        'dir_labels': dir_labels  # [B, H, W]
    }


def convert_predictions_to_boxes(predictions, batch_data, data_cfg):
    """将模型预测转换为边界框格式"""
    boxes = []

    batch_size = predictions['batch_size']
    pc_cfg = data_cfg['pointcloud']
    x_range = pc_cfg['x_range']
    y_range = pc_cfg['y_range']
    grid_h, grid_w = 32, 32

    for i in range(batch_size):
        # 解析预测结果
        cls_pred = predictions['cls_pred'][i]  # [C, H, W]
        reg_pred = predictions['reg_pred'][i]  # [7, H, W]

        # 使用阈值过滤
        cls_probs = torch.sigmoid(cls_pred)
        max_probs, max_indices = torch.max(cls_probs, dim=0)

        height, width = cls_probs.shape[1:]
        cell_size_x = (x_range[1] - x_range[0]) / grid_w
        cell_size_y = (y_range[1] - y_range[0]) / grid_h

        # 遍历BEV网格生成边界框
        for h in range(height):
            for w in range(width):
                if max_probs[h, w] > 0.3:  # 置信度阈值
                    class_id = max_indices[h, w].item()
                    reg_data = reg_pred[:, h, w]

                    class_names = ['Car', 'Cyclist', 'Truck']
                    if class_id >= len(class_names):
                        continue

                    # 将相对坐标转换为绝对坐标
                    abs_x = w * cell_size_x + x_range[0] + reg_data[0] * cell_size_x
                    abs_y = h * cell_size_y + y_range[0] + reg_data[1] * cell_size_y
                    abs_z = reg_data[2]

                    box = {
                        'class': class_names[class_id],
                        'bbox_3d': [
                            abs_x.item(),  # x
                            abs_y.item(),  # y
                            abs_z.item(),  # z
                            reg_data[5].item(),  # height
                            reg_data[3].item(),  # width
                            reg_data[4].item(),  # length
                            reg_data[6].item()  # rotation
                        ],
                        'score': max_probs[h, w].item()
                    }
                    boxes.append(box)

    return boxes


def convert_labels_to_boxes(batch_data):
    """将标签数据转换为边界框格式"""
    boxes = []

    batch_size = len(batch_data['labels'])

    for i in range(batch_size):
        if batch_data['labels'][i] is None:
            continue

        for label in batch_data['labels'][i]:
            box = {
                'class': label['type'],
                'bbox_3d': [
                    label['location'][0],  # x
                    label['location'][1],  # y
                    label['location'][2],  # z
                    label['dimensions'][0],  # height
                    label['dimensions'][1],  # width
                    label['dimensions'][2],  # length
                    label['rotation_y']  # rotation
                ]
            }
            boxes.append(box)

    return boxes


# collate_fn 和 move_to_device
def collate_fn(batch):
    """自定义批次整理函数"""
    if len(batch) == 0:
        return {}

    max_voxels = max(len(item['voxels']) for item in batch)
    feature_dim = batch[0]['voxels'].shape[-1] if len(batch[0]['voxels']) > 0 else 8

    padded_voxels = []
    padded_coords = []
    padded_numpoints = []

    for item in batch:
        voxels = item['voxels']
        coords = item['coordinates']
        numpoints = item['num_points']

        if len(voxels) < max_voxels:
            pad_size = max_voxels - len(voxels)
            padded_voxels.append(torch.cat([
                voxels,
                torch.zeros(pad_size, voxels.shape[1], feature_dim)
            ], dim=0))
            padded_coords.append(torch.cat([
                coords,
                torch.zeros(pad_size, 3, dtype=coords.dtype)
            ], dim=0))
            padded_numpoints.append(torch.cat([
                numpoints,
                torch.zeros(pad_size, dtype=numpoints.dtype)
            ], dim=0))
        else:
            padded_voxels.append(voxels[:max_voxels])
            padded_coords.append(coords[:max_voxels])
            padded_numpoints.append(numpoints[:max_voxels])

    return {
        'voxels': torch.stack(padded_voxels),
        'coordinates': torch.stack(padded_coords),
        'num_points': torch.stack(padded_numpoints),
        'image': torch.stack([item['image'] for item in batch]),
        'calib': [item['calib'] for item in batch],
        'sample_id': [item['sample_id'] for item in batch],
        'labels': [item.get('labels', []) for item in batch]
    }


def move_to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    elif hasattr(data, 'to'):
        try:
            return data.to(device)
        except:
            return data  # 如果移动失败，返回原数据
    else:
        return data  # 基本数据类型和自定义对象不移动


def train_model():
    # 加载配置
    with open('../configs/data.yaml', 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    with open('../configs/model.yaml', 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集
    train_dataset = DRadDataset(data_cfg, split='train')
    val_dataset = DRadDataset(data_cfg, split='val')

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # 减小批次大小避免内存问题
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 创建模型
    model = RadarCameraFusionModel(model_cfg['model']).to(device)

    # 损失函数和优化器
    criterion = WeightedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 评估指标
    metrics = DetectionMetrics()

    # 训练循环
    num_epochs = 10  # 先训练少量epoch测试
    best_val_loss = float('inf')
    best_mAP = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = move_to_device(batch_data, device)
            targets = prepare_targets(batch_data, data_cfg)
            targets = move_to_device(targets, device)

            optimizer.zero_grad()
            predictions = model(batch_data)
            losses = criterion(predictions, targets)

            losses['weighted_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += losses['weighted_loss'].item()
            pbar.set_postfix({'Loss': f"{losses['weighted_loss'].item():.4f}"})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        metrics.reset()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for batch_data in pbar:
                batch_data = move_to_device(batch_data, device)
                targets = prepare_targets(batch_data, data_cfg)
                targets = move_to_device(targets, device)
                predictions = model(batch_data)
                losses = criterion(predictions, targets)
                val_loss += losses['weighted_loss'].item()

                # 计算评估指标
                pred_boxes = convert_predictions_to_boxes(predictions, batch_data, data_cfg)
                gt_boxes = convert_labels_to_boxes(batch_data)
                metrics.add_batch(pred_boxes, gt_boxes)

                pbar.set_postfix({'Loss': f"{losses['weighted_loss'].item():.4f}"})

        # 计算平均损失和评估指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_results = metrics.evaluate()
        current_mAP = val_results['mAP']

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"验证集 mAP: {current_mAP:.4f}")
        metrics.print_summary()

        # 保存最佳模型（基于mAP）
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'mAP': best_mAP,
                'config': model_cfg
            }, '../checkpoints/best_model.pth')
            print(f"保存最佳模型，验证集 mAP: {best_mAP:.4f}")

        # 也保存基于损失的检查点
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'mAP': current_mAP,
                'config': model_cfg
            }, '../checkpoints/best_loss_model.pth')
            print(f"保存最佳损失模型，验证损失: {best_val_loss:.4f}")

        scheduler.step()


if __name__ == '__main__':
    os.makedirs('../checkpoints', exist_ok=True)
    train_model()