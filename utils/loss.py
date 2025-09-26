# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        predictions: dict containing:
            - cls_pred: [B, C, H, W] 分类预测
            - reg_pred: [B, 7, H, W] 回归预测 (x, y, z, w, l, h, theta)
            - dir_pred: [B, 2, H, W] 方向预测
        targets: dict containing:
            - cls_labels: [B, H, W] 分类标签
            - reg_targets: [B, H, W, 7] 回归目标
            - dir_labels: [B, H, W] 方向标签
        """
        cls_pred = predictions['cls_pred']  # [B, C, H, W]
        reg_pred = predictions['reg_pred']  # [B, 7, H, W]
        dir_pred = predictions['dir_pred']  # [B, 2, H, W]

        cls_labels = targets['cls_labels']  # [B, H, W]
        reg_targets = targets['reg_targets']  # [B, H, W, 7]
        dir_labels = targets['dir_labels']  # [B, H, W]

        # 调整形状以匹配
        B, C, H, W = cls_pred.shape
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(-1, 7)  # [B*H*W, 7]
        dir_pred = dir_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)  # [B*H*W, 2]

        cls_labels = cls_labels.reshape(-1)  # [B*H*W]
        reg_targets = reg_targets.reshape(-1, 7)  # [B*H*W, 7]
        dir_labels = dir_labels.reshape(-1)  # [B*H*W]

        # 分类损失 (Focal Loss)
        cls_loss = self.focal_loss(cls_pred, cls_labels)

        # 回归损失 (Smooth L1)
        reg_loss = self.smooth_l1_loss(reg_pred, reg_targets, cls_labels)

        # 方向损失
        dir_loss = self.direction_loss(dir_pred, dir_labels, cls_labels)

        total_loss = cls_loss + reg_loss + dir_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'dir_loss': dir_loss
        }

    def focal_loss(self, pred, target):
        """Focal Loss for classification"""
        # 确保目标在有效范围内
        valid_mask = (target >= 0) & (target < self.num_classes)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)

        # 只处理有效样本
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask].long()

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(pred_valid, target_valid, reduction='none')

        # 计算概率
        pt = torch.exp(-ce_loss)

        # Focal Loss权重
        focal_weight = (self.alpha * (1 - pt) ** self.gamma)

        # 应用Focal Loss
        focal_loss = (focal_weight * ce_loss).mean()

        return focal_loss

    def smooth_l1_loss(self, pred, target, labels):
        """Smooth L1 Loss for regression"""
        # 只计算正样本的回归损失
        pos_mask = labels > 0  # 正样本

        if not pos_mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred_pos = pred[pos_mask]
        target_pos = target[pos_mask]

        diff = torch.abs(pred_pos - target_pos)
        loss = torch.where(diff < 1, 0.5 * diff ** 2, diff - 0.5)

        # 对每个回归参数求平均损失
        reg_loss = loss.mean()

        return reg_loss

    def direction_loss(self, pred, target, labels):
        """方向分类损失"""
        # 只计算正样本的方向损失
        pos_mask = labels > 0

        if not pos_mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred_pos = pred[pos_mask]
        target_pos = target[pos_mask].long()

        dir_loss = F.cross_entropy(pred_pos, target_pos)
        return dir_loss


class WeightedLoss(nn.Module):
    """加权损失函数"""

    def __init__(self, weights={'cls': 1.0, 'reg': 2.0, 'dir': 0.2}):
        super().__init__()
        self.weights = weights
        self.detection_loss = DetectionLoss()

    def forward(self, predictions, targets):
        losses = self.detection_loss(predictions, targets)

        weighted_loss = (
                self.weights['cls'] * losses['cls_loss'] +
                self.weights['reg'] * losses['reg_loss'] +
                self.weights['dir'] * losses['dir_loss']
        )

        losses['weighted_loss'] = weighted_loss
        return losses