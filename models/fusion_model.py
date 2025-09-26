# models/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class VoxelBackBone8x(nn.Module):
    """体素特征提取网络，基于3D CNN架构"""

    def __init__(self, input_channels=13, output_channels=128):
        super().__init__()

        # 使用类似VoxelNet的架构
        self.conv_layers = nn.Sequential(
            # 第一层：扩大感受野
            nn.Conv3d(input_channels, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # 第二层：下采样
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # 残差块1
            self._make_residual_block(64, 64),

            # 第三层：下采样
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # 残差块2
            self._make_residual_block(128, 128),

            # 第四层：特征提取
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            # 残差块3
            self._make_residual_block(256, 256),

            # 最终投影层
            nn.Conv3d(256, output_channels, 1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )

        # 初始化权重
        self._initialize_weights()

    def _make_residual_block(self, in_channels, out_channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels)
        )

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 应用卷积层
        features = self.conv_layers(x)
        return features



class AnchorFreeHead(nn.Module):
    """无锚框检测头"""

    def __init__(self, input_channels=384, num_classes=3, num_attrs=10):
        super().__init__()
        self.num_classes = num_classes

        # 共享特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 分类分支
        self.cls_conv = nn.Conv2d(256, num_classes, 1)

        # 回归分支 (x, y, z, w, l, h, theta)
        self.reg_conv = nn.Conv2d(256, 7, 1)

        # 方向分支
        self.dir_conv = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        shared_feat = self.shared_conv(x)

        cls_pred = self.cls_conv(shared_feat)  # [B, num_classes, H, W]
        reg_pred = self.reg_conv(shared_feat)  # [B, 7, H, W]
        dir_pred = self.dir_conv(shared_feat)  # [B, 2, H, W]

        return cls_pred, reg_pred, dir_pred


class RadarCameraFusionModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.target_bev_size = (32, 32)  # 统一定义BEV尺寸

        # 点云处理分支
        self.pointcloud_backbone = VoxelBackBone8x(
            input_channels=model_cfg['pointcloud']['input_channels'],
            output_channels=model_cfg['pointcloud']['output_channels']
        )

        # 图像处理分支
        self.image_backbone = self.build_image_backbone()

        # 特征融合模块
        self.fusion_module = BEVFusionModule(
            pointcloud_channels=model_cfg['pointcloud']['output_channels'],
            image_channels=512,
            output_channels=model_cfg['fusion']['fusion_channels'],
            target_bev_size=self.target_bev_size  # 传入目标尺寸
        )

        # 3D检测头
        self.detection_head = AnchorFreeHead(
            input_channels=model_cfg['fusion']['fusion_channels'],
            num_classes=model_cfg['detection_head']['num_classes']
        )

    def build_image_backbone(self):
        """构建图像特征提取网络"""
        cfg = self.model_cfg['image']
        backbone = models.resnet18(weights=cfg['pretrained'])

        # 移除最后的全连接层和平均池化层
        modules = list(backbone.children())[:-2]
        return nn.Sequential(*modules)

    def forward(self, batch_dict):
        # 点云特征提取
        voxel_features = self.extract_pointcloud_features(batch_dict)

        # 图像特征提取
        image_features = self.extract_image_features(batch_dict)

        # 特征融合
        fused_features = self.fusion_module(voxel_features, image_features)

        # 3D检测
        cls_pred, reg_pred, dir_pred = self.detection_head(fused_features)

        # 格式化输出
        predictions = self.format_predictions(
            cls_pred, reg_pred, dir_pred, batch_dict
        )

        return predictions

    def extract_pointcloud_features(self, batch_dict):
        """提取点云特征"""
        voxels = batch_dict['voxels']  # [B, max_voxels, max_points, feature_dim]
        coordinates = batch_dict['coordinates']  # [B, max_voxels, 3]
        num_points = batch_dict['num_points']  # [B, max_voxels]

        batch_size = voxels.shape[0]
        batch_features = []

        target_bev_size = (32, 32)  # 固定BEV特征图尺寸

        for i in range(batch_size):
            # 创建体素网格
            voxel_grid = self.create_voxel_grid(
                voxels[i], coordinates[i], num_points[i]
            )

            if voxel_grid is not None:
                # 3D卷积提取特征
                voxel_features = self.pointcloud_backbone(voxel_grid.unsqueeze(0))

                # 转换为BEV特征 (沿z轴最大池化)
                bev_features = torch.max(voxel_features, dim=2)[0]  # [1, C, H, W]

                # 调整到统一尺寸
                if bev_features.shape[-2:] != target_bev_size:
                    bev_features = F.interpolate(
                        bev_features, size=target_bev_size,
                        mode='bilinear', align_corners=False
                    )

                batch_features.append(bev_features)
            else:
                # 创建统一尺寸的空特征图
                empty_feat = torch.zeros(
                    1, self.model_cfg['pointcloud']['output_channels'],
                    target_bev_size[0], target_bev_size[1],
                    device=voxels.device
                )
                batch_features.append(empty_feat)

        return torch.cat(batch_features, dim=0)  # [B, C, H, W]

    def create_voxel_grid(self, voxels, coordinates, num_points):
        """创建密集体素网格"""
        if len(voxels) == 0:
            return None

        # 使用配置中的范围计算网格尺寸
        pc_cfg = self.model_cfg.get('pointcloud', {})
        x_range = pc_cfg.get('x_range', [0, 25])
        y_range = pc_cfg.get('y_range', [-25, 25])
        z_range = pc_cfg.get('z_range', [-2, 5])
        voxel_size = pc_cfg.get('voxel_size', [0.16, 0.16, 0.5])

        # 计算网格尺寸
        grid_size_x = int(np.ceil((x_range[1] - x_range[0]) / voxel_size[0]))
        grid_size_y = int(np.ceil((y_range[1] - y_range[0]) / voxel_size[1]))
        grid_size_z = int(np.ceil((z_range[1] - z_range[0]) / voxel_size[2]))

        grid_size = [grid_size_x, grid_size_y, grid_size_z]

        feature_dim = voxels.shape[-1]

        # 创建空网格
        voxel_grid = torch.zeros(
            feature_dim, grid_size[2], grid_size[1], grid_size[0],
            device=voxels.device
        )

        # 过滤有效体素坐标
        valid_mask = (
                (coordinates[:, 0] >= 0) & (coordinates[:, 0] < grid_size[0]) &
                (coordinates[:, 1] >= 0) & (coordinates[:, 1] < grid_size[1]) &
                (coordinates[:, 2] >= 0) & (coordinates[:, 2] < grid_size[2])
        )

        if not valid_mask.any():
            return None

        valid_coords = coordinates[valid_mask]
        valid_voxels = voxels[valid_mask]
        valid_num_points = num_points[valid_mask]

        # 填充体素网格
        for i, (x, y, z) in enumerate(valid_coords):
            voxel_data = valid_voxels[i][:valid_num_points[i]]
            if len(voxel_data) > 0:
                # 使用平均特征
                mean_feature = voxel_data.mean(dim=0)  # [feature_dim]
                voxel_grid[:, z, y, x] = mean_feature

        return voxel_grid

    def extract_image_features(self, batch_dict):
        """提取图像特征"""
        images = batch_dict['image']  # [B, 3, H, W]
        features = self.image_backbone(images)  # [B, 512, H/32, W/32]
        return features

    def format_predictions(self, cls_pred, reg_pred, dir_pred, batch_dict):
        """格式化预测结果"""
        batch_size = cls_pred.shape[0]

        predictions = {
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'dir_pred': dir_pred,
            'batch_size': batch_size,
            'sample_ids': batch_dict['sample_id']
        }

        return predictions


# BEV融合模块
class BEVFusionModule(nn.Module):
    """BEV空间特征融合模块"""

    def __init__(self, pointcloud_channels, image_channels, output_channels, target_bev_size=(32, 32)):
        super().__init__()
        self.target_bev_size = target_bev_size

        # 点云特征投影
        self.pointcloud_projection = nn.Sequential(
            nn.Conv2d(pointcloud_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1)
        )

        # 图像特征投影和上采样
        self.image_projection = nn.Sequential(
            nn.Conv2d(image_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(size=target_bev_size, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1)
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, pointcloud_features, image_features):
        # 确保点云特征图尺寸正确
        if pointcloud_features.shape[-2:] != self.target_bev_size:
            pointcloud_features = F.interpolate(
                pointcloud_features, size=self.target_bev_size,
                mode='bilinear', align_corners=False
            )

        # 特征投影
        pc_bev = self.pointcloud_projection(pointcloud_features)
        img_bev = self.image_projection(image_features)

        # 特征拼接和融合
        fused = torch.cat([pc_bev, img_bev], dim=1)
        return self.fusion_conv(fused)