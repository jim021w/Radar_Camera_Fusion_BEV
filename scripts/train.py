# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
from tqdm import tqdm
import argparse
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(project_root)

from data import DRadDataset
from models.fusion_model import RadarCameraFusionModel
from utils import DetectionMetrics, WeightedLoss, RotatedNMS3D


class UnifiedRadarCameraModel:
    """统一的雷达-相机融合模型训练和推理类"""

    def __init__(self, config_dir="../configs"):
        self.config_dir = config_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载配置
        self.load_configs()

        # 初始化模型和组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = WeightedLoss()
        self.metrics = DetectionMetrics()
        self.nms = RotatedNMS3D(iou_threshold=0.5, score_threshold=0.1)

    def load_configs(self):
        """加载配置文件"""
        with open(os.path.join(self.config_dir, 'data.yaml'), 'r', encoding='utf-8') as f:
            self.data_cfg = yaml.safe_load(f)

        with open(os.path.join(self.config_dir, 'model.yaml'), 'r', encoding='utf-8') as f:
            self.model_cfg = yaml.safe_load(f)

    def setup_model(self, checkpoint_path=None):
        """设置模型，可选加载检查点"""
        self.model = RadarCameraFusionModel(self.model_cfg['model']).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型检查点: {checkpoint_path}")
            if 'mAP' in checkpoint:
                print(f"训练时最佳 mAP: {checkpoint['mAP']:.4f}")
        else:
            print("使用随机初始化权重")

    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def prepare_targets(self, batch_data):
        """目标准备函数"""
        batch_size = batch_data['voxels'].shape[0]
        pc_cfg = self.data_cfg['pointcloud']
        x_range = pc_cfg['x_range']
        y_range = pc_cfg['y_range']

        # BEV网格尺寸
        grid_h, grid_w = 32, 32
        cell_size_x = (x_range[1] - x_range[0]) / grid_w
        cell_size_y = (y_range[1] - y_range[0]) / grid_h

        # 初始化标签
        cls_labels = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.long)
        reg_targets = torch.zeros(batch_size, grid_h, grid_w, 7)
        dir_labels = torch.zeros(batch_size, grid_h, grid_w, dtype=torch.long)

        # 类别映射
        class_map = {'Car': 0, 'Cyclist': 1, 'Truck': 2}

        for i in range(batch_size):
            if 'labels' in batch_data and batch_data['labels'][i]:
                for label in batch_data['labels'][i]:
                    center_x, center_y, center_z = label['location']
                    height, width, length = label['dimensions']
                    rotation = label['rotation_y']

                    # 计算网格坐标
                    grid_x = int((center_x - x_range[0] + cell_size_x / 2) / cell_size_x)
                    grid_y = int((center_y - y_range[0] + cell_size_y / 2) / cell_size_y)
                    grid_x = max(0, min(grid_w - 1, grid_x))
                    grid_y = max(0, min(grid_h - 1, grid_y))

                    if label['type'] not in class_map:
                        continue

                    class_id = class_map[label['type']]
                    cls_labels[i, grid_y, grid_x] = class_id + 1  # 0保留给背景

                    # 计算回归目标
                    center_x_in_voxel = (center_x - x_range[0]) % cell_size_x
                    center_y_in_voxel = (center_y - y_range[0]) % cell_size_y

                    reg_targets[i, grid_y, grid_x, 0] = (center_x_in_voxel - cell_size_x / 2) / cell_size_x
                    reg_targets[i, grid_y, grid_x, 1] = (center_y_in_voxel - cell_size_y / 2) / cell_size_y
                    reg_targets[i, grid_y, grid_x, 2] = center_z
                    reg_targets[i, grid_y, grid_x, 3] = torch.log(torch.tensor(height + 1e-8))
                    reg_targets[i, grid_y, grid_x, 4] = torch.log(torch.tensor(width + 1e-8))
                    reg_targets[i, grid_y, grid_x, 5] = torch.log(torch.tensor(length + 1e-8))

                    rotation = rotation % (2 * torch.pi)
                    if rotation > torch.pi:
                        rotation -= 2 * torch.pi
                    reg_targets[i, grid_y, grid_x, 6] = rotation

                    dir_labels[i, grid_y, grid_x] = 0 if abs(rotation) < torch.pi / 2 else 1

        reg_targets = reg_targets.permute(0, 3, 1, 2)

        return {
            'cls_labels': cls_labels,
            'reg_targets': reg_targets,
            'dir_labels': dir_labels
        }

    def convert_predictions_to_boxes(self, predictions, batch_data, max_predictions_per_sample=25):
        """使用NMS将预测转换为边界框"""
        boxes = []
        batch_size = predictions['batch_size']
        pc_cfg = self.data_cfg['pointcloud']
        x_range = pc_cfg['x_range']
        y_range = pc_cfg['y_range']
        grid_h, grid_w = 32, 32

        for i in range(batch_size):
            cls_pred = predictions['cls_pred'][i]
            reg_pred = predictions['reg_pred'][i]

            cls_probs = torch.sigmoid(cls_pred)
            max_probs, max_indices = torch.max(cls_probs, dim=0)

            height, width = cls_probs.shape[1:]
            cell_size_x = (x_range[1] - x_range[0]) / grid_w
            cell_size_y = (y_range[1] - y_range[0]) / grid_h

            # 收集候选框
            candidate_boxes = []
            candidate_scores = []
            candidate_classes = []

            # 向量化操作
            all_h, all_w = torch.meshgrid(
                torch.arange(height, device=cls_probs.device),
                torch.arange(width, device=cls_probs.device),
                indexing='ij'
            )
            all_h = all_h.flatten()
            all_w = all_w.flatten()
            all_probs = max_probs.flatten()
            all_indices = max_indices.flatten()

            # 应用置信度阈值
            confidence_threshold = 0.1
            valid_mask = all_probs > confidence_threshold
            valid_h = all_h[valid_mask]
            valid_w = all_w[valid_mask]
            valid_probs = all_probs[valid_mask]
            valid_indices = all_indices[valid_mask]

            if len(valid_probs) == 0:
                continue

            # 限制最大预测数量
            if len(valid_probs) > max_predictions_per_sample:
                topk_indices = torch.topk(valid_probs, max_predictions_per_sample).indices
                valid_h = valid_h[topk_indices]
                valid_w = valid_w[topk_indices]
                valid_probs = valid_probs[topk_indices]
                valid_indices = valid_indices[topk_indices]

            # 处理有效预测
            for idx in range(len(valid_probs)):
                h_idx = valid_h[idx].item()
                w_idx = valid_w[idx].item()
                class_id = valid_indices[idx].item()
                prob = valid_probs[idx].item()

                reg_data = reg_pred[:, h_idx, w_idx]

                # 转换为绝对坐标
                abs_x = w_idx * cell_size_x + x_range[0] + reg_data[0] * cell_size_x
                abs_y = h_idx * cell_size_y + y_range[0] + reg_data[1] * cell_size_y
                abs_z = reg_data[2]

                # 尺寸反变换
                width_3d = torch.exp(reg_data[3]).item()
                length_3d = torch.exp(reg_data[4]).item()
                height_3d = torch.exp(reg_data[5]).item()

                box_3d = torch.tensor([
                    abs_x.item(), abs_y.item(), abs_z.item(),
                    width_3d, length_3d, height_3d, reg_data[6].item()
                ], device=reg_pred.device)

                candidate_boxes.append(box_3d)
                candidate_scores.append(prob)
                candidate_classes.append(class_id)

            if len(candidate_boxes) == 0:
                continue

            # 应用NMS
            candidate_boxes = torch.stack(candidate_boxes)
            candidate_scores = torch.tensor(candidate_scores, device=candidate_boxes.device)

            keep_indices = self.nms(candidate_boxes, candidate_scores)

            # 限制NMS后的预测数量
            if len(keep_indices) > max_predictions_per_sample:
                nms_scores = candidate_scores[keep_indices]
                top_nms_indices = torch.topk(nms_scores, max_predictions_per_sample).indices
                keep_indices = keep_indices[top_nms_indices]

            # 保留NMS后的框
            class_names = ['Car', 'Cyclist', 'Truck']
            for idx in keep_indices:
                class_id = candidate_classes[idx]
                if class_id >= len(class_names):
                    continue

                box = {
                    'class': class_names[class_id],
                    'bbox_3d': candidate_boxes[idx].tolist(),
                    'score': candidate_scores[idx].item()
                }
                boxes.append(box)

        return boxes

    def convert_labels_to_boxes(self, batch_data):
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
                        label['location'][0],
                        label['location'][1],
                        label['location'][2],
                        label['dimensions'][0],
                        label['dimensions'][1],
                        label['dimensions'][2],
                        label['rotation_y']
                    ]
                }
                boxes.append(box)

        return boxes

    def collate_fn(self, batch):
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

    def move_to_device(self, data):
        """将数据移动到指定设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.move_to_device(item) for item in data]
        else:
            return data

    def train(self, num_epochs=10, batch_size=2, checkpoint_dir="../checkpoints"):
        """训练模型"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 设置模型和优化器
        self.setup_model()
        self.setup_optimizer()

        # 创建数据集
        train_dataset = DRadDataset(self.data_cfg, split='train')
        val_dataset = DRadDataset(self.data_cfg, split='val')

        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn
        )

        best_val_loss = float('inf')
        best_mAP = 0.0

        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
            for batch_idx, batch_data in enumerate(pbar):
                batch_data = self.move_to_device(batch_data)
                targets = self.prepare_targets(batch_data)
                targets = self.move_to_device(targets)

                self.optimizer.zero_grad()
                predictions = self.model(batch_data)
                losses = self.criterion(predictions, targets)

                losses['weighted_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += losses['weighted_loss'].item()
                pbar.set_postfix({'Loss': f"{losses['weighted_loss'].item():.4f}"})

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            self.metrics.reset()

            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
                for batch_data in pbar:
                    batch_data = self.move_to_device(batch_data)
                    targets = self.prepare_targets(batch_data)
                    targets = self.move_to_device(targets)
                    predictions = self.model(batch_data)
                    losses = self.criterion(predictions, targets)
                    val_loss += losses['weighted_loss'].item()

                    # 计算评估指标
                    pred_boxes = self.convert_predictions_to_boxes(predictions, batch_data)
                    gt_boxes = self.convert_labels_to_boxes(batch_data)
                    self.metrics.add_batch(pred_boxes, gt_boxes)

                    pbar.set_postfix({'Loss': f"{losses['weighted_loss'].item():.4f}"})

            # 计算平均损失和评估指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_results = self.metrics.evaluate()
            current_mAP = val_results['mAP']

            print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"验证集 mAP: {current_mAP:.4f}")
            self.metrics.print_summary()

            # 保存最佳模型
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'mAP': best_mAP,
                    'config': self.model_cfg
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"保存最佳模型，验证集 mAP: {best_mAP:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                    'mAP': current_mAP,
                    'config': self.model_cfg
                }, os.path.join(checkpoint_dir, 'best_loss_model.pth'))
                print(f"保存最佳损失模型，验证损失: {best_val_loss:.4f}")

            self.scheduler.step()

    def inference(self, checkpoint_path, output_dir="../results", batch_size=1):
        """推理模式"""
        os.makedirs(output_dir, exist_ok=True)

        # 设置模型
        self.setup_model(checkpoint_path)
        self.model.eval()

        # 创建测试数据集
        test_dataset = DRadDataset(self.data_cfg, split='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        # 推理和评估
        all_predictions = []
        all_ground_truths = []
        self.metrics.reset()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch_data = self.move_to_device(batch_data)
                predictions = self.model(batch_data)

                # 收集预测和真值
                pred_boxes = self.convert_predictions_to_boxes(predictions, batch_data)
                gt_boxes = self.convert_labels_to_boxes(batch_data)

                all_predictions.extend(pred_boxes)
                all_ground_truths.extend(gt_boxes)

                # 添加到评估器
                self.metrics.add_batch(pred_boxes, gt_boxes)

                if (batch_idx + 1) % 10 == 0:
                    print(f"已处理 {batch_idx + 1} 个批次")

        # 计算评估指标
        print("\n开始计算测试集评估指标...")
        test_results = self.metrics.evaluate()
        self.metrics.print_summary()

        # 保存评估结果
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("3D目标检测测试集评估结果\n")
            f.write("=" * 50 + "\n")
            for cls in self.metrics.classes:
                result = test_results[cls]
                f.write(f"{cls}: AP={result['ap']:.4f}, Precision={result['precision']:.4f}, "
                        f"Recall={result['recall']:.4f}, F1={result['f1']:.4f}\n")
            f.write(f"mAP: {test_results['mAP']:.4f}\n")

        print(f"推理完成，评估结果已保存到: {output_dir}")

        return test_results


def main():
    parser = argparse.ArgumentParser(description="雷达-相机融合3D目标检测")
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'inference'],
                        help='运行模式: train 或 inference')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/best_model.pth",
                        help='模型检查点路径（推理模式需要）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数（训练模式）')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    parser.add_argument('--output_dir', type=str, default="../results",
                        help='输出目录（推理模式）')
    parser.add_argument('--config_dir', type=str, default="../configs",
                        help='配置文件目录')

    args = parser.parse_args()

    # 创建模型实例
    model_manager = UnifiedRadarCameraModel(config_dir=args.config_dir)

    if args.mode == 'train':
        print("开始训练模式...")
        model_manager.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.mode == 'inference':
        print("开始推理模式...")
        if not os.path.exists(args.checkpoint):
            print(f"警告: 检查点文件 {args.checkpoint} 不存在")
            return

        model_manager.inference(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()