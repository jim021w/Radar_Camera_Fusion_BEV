# data/dataset.py


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .calibration import Calibration
from .preprocess import PointCloudProcessor, ImageProcessor


class DRadDataset(Dataset):
    """DRadDataset数据集类（支持4D雷达特征）"""

    def __init__(self, cfg, split='train', transform=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform

        # 获取数据根目录
        self.root_path = cfg['data']['root_path']

        # 读取划分文件
        split_file = os.path.join(self.root_path, 'ImageSets', cfg['data'].get(f'{split}_split', f'{split}.txt'))

        with open(split_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines() if line.strip()]

        # 初始化处理器
        self.calib = Calibration(cfg=cfg)
        self.pc_processor = PointCloudProcessor(cfg)
        self.img_processor = ImageProcessor(cfg)
        self.calib_cache = {}
        self.max_cache_size = 1000  # 缓存大小限制


    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        if sample_id in self.calib_cache:
            return self.calib_cache[sample_id]

        # 确定数据目录
        base_dir = 'testing' if self.split == 'test' else 'training'

        # 加载标定数据（先加载标定，因为点云投影需要）
        calib_path = os.path.join(self.root_path, base_dir, 'calib', f'{sample_id}.txt')
        success = self.calib.load_calibration(calib_path)  # 返回加载是否成功
        if not success:
            raise ValueError(f"标定文件加载失败: {calib_path}")
        calib_data = self.calib  # 使用Calibration对象本身

        # 加载点云数据（4D雷达数据）
        pc_path = os.path.join(self.root_path, base_dir, 'velodyne', f'{sample_id}.bin')
        point_cloud = self.load_pointcloud(pc_path)

        # 加载图像数据
        img_path = os.path.join(self.root_path, base_dir, 'image_2', f'{sample_id}.png')
        image = self.img_processor.load_image(img_path)
        if image is None:
            raise FileNotFoundError(f"图像文件加载失败: {img_path}")

        # 加载标签（测试集没有标签）
        labels = None
        if self.split != 'test':
            label_path = os.path.join(self.root_path, base_dir, 'label_2', f'{sample_id}.txt')
            labels = self.load_labels(label_path)

        # 数据增强
        if self.transform and labels is not None:
            point_cloud, image, labels = self.transform(point_cloud, image, labels)
        elif self.transform:
            point_cloud, image, _ = self.transform(point_cloud, image, None)

        # 处理点云
        voxels, coordinates, num_points = self.pc_processor.process(point_cloud)

        # 处理图像
        image = self.img_processor.process(image)

        # 转换为Tensor
        sample = {
            'voxels': torch.from_numpy(voxels).float(),
            'coordinates': torch.from_numpy(coordinates).int(),
            'num_points': torch.from_numpy(num_points).int(),
            'image': torch.from_numpy(image).float(),
            'calib': calib_data,
            'sample_id': sample_id
        }

        if labels is not None:
            sample['labels'] = labels

        self.calib_cache[sample_id]=sample

        if len(self.calib_cache) > self.max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.calib_cache))
            del self.calib_cache[oldest_key]

        return sample

    def load_pointcloud(self, pc_path):
        """加载4D雷达点云数据"""
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Point cloud file not found: {pc_path}")

        # 读取二进制数据
        data = np.fromfile(pc_path, dtype=np.float32)

        # 根据数据维度调整形状
        # 4D雷达数据格式: [距离, 方位角, 俯仰角, 多普勒, 功率, x, y, z]
        if len(data) % 8 != 0:
            raise ValueError(f"点云数据格式错误，文件: {pc_path}, 数据长度: {len(data)}")
        else:
            point_cloud = data.reshape(-1, 8)

        return point_cloud

    def load_labels(self, label_path):
        """加载KITTI格式标签数据"""
        labels = []
        if not os.path.exists(label_path):
            return labels

        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue

                data = line.split()
                if len(data) < 15:
                    continue

                # 解析KITTI格式标签
                label = {
                    'type': data[0],
                    'truncated': float(data[1]),
                    'occluded': int(data[2]),
                    'alpha': float(data[3]),
                    'bbox': [float(x) for x in data[4:8]],  # left, top, right, bottom
                    'dimensions': [float(data[8]), float(data[9]), float(data[10])],  # height, width, length
                    'location': [float(data[11]), float(data[12]), float(data[13])],  # x, y, z
                    'rotation_y': float(data[14]),
                }

                # 添加置信度（如果存在）
                if len(data) > 15:
                    label['score'] = float(data[15])
                else:
                    label['score'] = 1.0  # 真值标签置信度为1

                # 过滤类别
                valid_classes = self.cfg['data'].get('classes', ['Car', 'Cyclist', 'Truck'])
                if label['type'] in valid_classes:
                    labels.append(label)

        return labels