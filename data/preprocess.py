# data/preprocess.py

import numpy as np
import cv2

class PointCloudProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        pc_cfg = cfg['pointcloud']

        self.x_range = pc_cfg['x_range']
        self.y_range = pc_cfg['y_range']
        self.z_range = pc_cfg['z_range']
        self.voxel_size = pc_cfg['voxel_size']
        self.max_points_per_voxel = pc_cfg['max_points_per_voxel']
        self.max_voxels = pc_cfg['max_voxels']
        self.min_points = pc_cfg.get('min_points', 1)
        self.max_points = pc_cfg.get('max_points', 50000)
        self.use_intensity = pc_cfg.get('use_intensity', True)
        self.use_velocity = pc_cfg.get('use_velocity', True)
        self.use_power = pc_cfg.get('use_power', True)

    def process(self, point_cloud):
        # 先进行特征提取和过滤
        point_cloud = self.filter_and_extract_features(point_cloud)

        if len(point_cloud) == 0:
            # 返回空数据的正确形状
            feature_dim = 8  # 根据实际特征维度调整
            return (np.zeros((0, self.max_points_per_voxel, feature_dim), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32))

        # 再进行体素化
        voxels, coordinates, num_points = self.voxelization(point_cloud)
        return voxels, coordinates, num_points

    def voxelization(self, point_cloud):
        """体素化函数"""
        if len(point_cloud) == 0:
            feature_dim = 13  # 固定特征维度

            return (np.zeros((0, self.max_points_per_voxel, feature_dim), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32))

        # 计算体素坐标
        points_xyz = point_cloud[:, -3:]  # x,y,z坐标在最后3列

        voxel_coords = ((points_xyz - np.array([self.x_range[0], self.y_range[0], self.z_range[0]])) /
                        np.array(self.voxel_size)).astype(np.int32)

        # 过滤无效体素
        grid_size = (
            int((self.x_range[1] - self.x_range[0]) / self.voxel_size[0]),
            int((self.y_range[1] - self.y_range[0]) / self.voxel_size[1]),
            int((self.z_range[1] - self.z_range[0]) / self.voxel_size[2])
        )

        valid_mask = (
                (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < grid_size[0]) &
                (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < grid_size[1]) &
                (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < grid_size[2])
        )

        voxel_coords = voxel_coords[valid_mask]
        point_cloud = point_cloud[valid_mask]

        if len(point_cloud) == 0:
            feature_dim = point_cloud.shape[1] if len(point_cloud.shape) > 1 else 3
            return (np.zeros((0, self.max_points_per_voxel, feature_dim), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32))

        # 生成体素ID（单样本处理）
        voxel_ids = (voxel_coords[:, 0] * (grid_size[1] * grid_size[2]) +
                     voxel_coords[:, 1] * grid_size[2] +
                     voxel_coords[:, 2])

        # 排序并分组
        sorted_idx = np.argsort(voxel_ids)
        voxel_ids = voxel_ids[sorted_idx]
        point_cloud = point_cloud[sorted_idx]
        unique_ids, counts = np.unique(voxel_ids, return_counts=True)

        # 限制最大体素数
        if len(unique_ids) > self.max_voxels:
            selected = np.random.choice(len(unique_ids), self.max_voxels, replace=False)
            unique_ids = unique_ids[selected]
            counts = counts[selected]

        # 向量化填充体素
        voxels = np.zeros((len(unique_ids), self.max_points_per_voxel, point_cloud.shape[1]), dtype=np.float32)
        coords = np.zeros((len(unique_ids), 3), dtype=np.int32)  # (x, y, z)
        num_points = np.minimum(counts, self.max_points_per_voxel)

        # 计算每个体素的起始索引
        cum_counts = np.cumsum(np.insert(counts, 0, 0))

        for i in range(len(unique_ids)):
            start_idx = cum_counts[i]
            end_idx = cum_counts[i + 1]
            points = point_cloud[start_idx:end_idx]

            if len(points) > self.max_points_per_voxel:
                points = points[np.random.choice(len(points), self.max_points_per_voxel, replace=False)]

            voxels[i, :len(points)] = points
            coords[i] = voxel_coords[sorted_idx[start_idx]]

        return voxels, coords, num_points

    def filter_and_extract_features(self, pointcloud):
        """对点云数据进行过滤和特征提取"""
        if len(pointcloud) == 0 :
            return np.array([])

        config = self.cfg.get('pointcloud', {})

        # 1. 数据过滤
        distance = pointcloud[:, 0]
        azimuth = pointcloud[:, 1]
        elevation = pointcloud[:, 2]
        doppler = pointcloud[:, 3]
        power = pointcloud[:, 4]
        x, y, z = pointcloud[:, 5], pointcloud[:, 6], pointcloud[:, 7]

        # 创建过滤掩码
        valid_mask = (
                (distance >= config.get('min_distance', 0)) &
                (distance <= config.get('max_distance', 25.0)) &
                (azimuth >= config.get('min_azimuth', -56.5)) &
                (azimuth <= config.get('max_azimuth', 56.5)) &
                (elevation >= config.get('min_elevation', -2.5)) &
                (elevation <= config.get('max_elevation', 22.5)) &
                (doppler >= config.get('min_doppler', -50.0)) &
                (doppler <= config.get('max_doppler', 50.0)) &
                (power >= config.get('min_power', -30.0)) &
                (power <= config.get('max_power', 10.0))
        )

        filtered_points = pointcloud[valid_mask]

        if len(filtered_points) == 0:
            return np.zeros((0, 13), dtype=np.float32)  # 固定13维特征

        # 2. 向量化特征提取（替代循环）
        features_list = []

        # 基础特征 (5维)
        base_features = filtered_points[:, :5]  # 距离,方位角,俯仰角,多普勒,功率

        # 坐标特征 (3维)
        coord_features = filtered_points[:, 5:8]  # x, y, z

        # 衍生特征 (5维) - 向量化计算
        range_squared = filtered_points[:, 0] ** 2
        speed_abs = np.abs(filtered_points[:, 3])
        intensity = filtered_points[:, 4] / (filtered_points[:, 0] + 1e-6)
        horizontal_distance = np.sqrt(filtered_points[:, 5] ** 2 + filtered_points[:, 6] ** 2)
        spatial_angle = np.arctan2(filtered_points[:, 6], filtered_points[:, 5])
        spatial_angle = np.nan_to_num(spatial_angle)  # 处理除零情况

        derived_features = np.column_stack([
            range_squared, speed_abs, intensity, horizontal_distance, spatial_angle
        ])

        # 根据配置选择特征
        if config.get('use_intensity', True) and config.get('use_derived_features', True):
            features = np.column_stack([base_features, coord_features, derived_features])
        elif config.get('use_intensity', True):
            features = np.column_stack([base_features, coord_features])
        else:
            features = coord_features

        return features.astype(np.float32)


class ImageProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        img_cfg = cfg.get('image', {})

        # 图像处理参数
        self.target_size = img_cfg.get('target_size', (640, 480))  # 目标尺寸 (宽, 高)
        self.normalize = img_cfg.get('normalize', True)
        self.mean = np.array(img_cfg.get('mean', [0.485, 0.456, 0.406]))
        self.std = np.array(img_cfg.get('std', [0.229, 0.224, 0.225]))
        self.keep_ratio = img_cfg.get('keep_ratio', False)
        self.color_mode = img_cfg.get('color_mode', 'rgb')  # 'rgb' 或 'bgr'

    def load_image(self, img_path):
        """加载图像文件"""
        try:
            # 读取图像 (OpenCV默认是BGR格式)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {img_path}")

            # 转换为RGB格式（如果配置要求）
            if self.color_mode == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image
        except Exception as e:
            print(f"图像加载错误 {img_path}: {e}")
            return None

    def process(self, image):
        """处理单张图像"""
        if image is None:
            return self._create_empty_image()

        # 调整尺寸
        image = self.resize_image(image)

        # 归一化
        if self.normalize:
            image = self.normalize_image(image)

        # 转换通道顺序 (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return image.astype(np.float32)

    def resize_image(self, image):
        """调整图像尺寸"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        if self.keep_ratio:
            # 保持宽高比调整
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 调整尺寸
            image = cv2.resize(image, (new_w, new_h))

            # 填充到目标尺寸
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # 使用边缘填充
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            # 直接调整到目标尺寸
            image = cv2.resize(image, (target_w, target_h))

        return image

    def normalize_image(self, image):
        """归一化图像"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # 标准化
        if self.normalize:
            image = (image - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)

        return image

    def _create_empty_image(self):
        """创建空图像"""
        target_w, target_h = self.target_size
        channels = 3  # RGB图像

        if self.normalize:
            empty_image = np.zeros((channels, target_h, target_w), dtype=np.float32)
        else:
            empty_image = np.zeros((channels, target_h, target_w), dtype=np.uint8)

        return empty_image

    def denormalize(self, image):
        """反归一化（用于可视化）"""
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        if self.normalize:
            image = image * self.std.reshape(1, 1, 3) + self.mean.reshape(1, 1, 3)
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image

    def project_points_to_image(self, points_3d, calib, camera_id=0):
        """将3D点投影到图像平面（用于可视化）"""
        if len(points_3d) == 0:
            return np.array([])

        # 确保只使用x,y,z坐标（最后3列）
        if points_3d.shape[1] > 3:
            points_xyz = points_3d[:, -3:]  # 提取x,y,z
        else:
            points_xyz = points_3d


        # 使用标定参数投影点云
        points_2d = calib.project_velo_to_image(points_xyz, camera_id)

        # 过滤在图像边界外的点
        if points_2d is not None:
            h, w = self.target_size[1], self.target_size[0]
            valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                         (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
            points_2d = points_2d[valid_mask]

        return points_2d