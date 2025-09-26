# data/transforms.py

import numpy as np
import random


class TrainTransform:
    """训练数据增强"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, point_cloud, image, labels):
        # 不对点云进行增强
        point_cloud, labels = point_cloud, labels

        # 图像增强
        image = self.augment_image(image)

        return point_cloud, image, labels

    def augment_image(self, image):
        """图像数据增强"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        image = np.clip(image, 0, 1)

        # 随机亮度调整
        if random.random() < 0.5:
            delta = random.uniform(-0.1, 0.1)
            image = np.clip(image + delta, 0, 1)

        # 随机对比度调整
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.1, 0.1)
            image = np.clip(alpha * (image - 0.5) + 0.5, 0, 1)

        # 随机高斯噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.005, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        return image


class EvalTransform:
    """评估数据变换（无数据增强）"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, point_cloud, image, labels):
        # 评估时不进行数据增强
        return point_cloud, image, labels